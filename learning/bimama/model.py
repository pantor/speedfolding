import numpy as np
import torch
from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead


class BiMamaNet(nn.Module):
    def __init__(self, num_rotations):
        super(BiMamaNet, self).__init__()
        self.num_rotations = num_rotations
        self.image_size = (256, 192)
        self.num_rotations = 20
        self.embedding_size = 8

        decoder_channels = (256, 192, 128, 96, 64)

        self.encoder = smp.encoders.get_encoder("resnext50_32x4d", in_channels=2, depth=5, weights=None)

        self.fling_decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5)
        self.fling_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=self.num_rotations + self.embedding_size)

        self.fling_to_fold_sleeve_decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5)
        self.fling_to_fold_sleeve_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=self.num_rotations + self.embedding_size)

        self.fling_to_fold_bottom_decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5)
        self.fling_to_fold_bottom_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=self.num_rotations + self.embedding_size)

        self.pick_decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5)
        self.pick_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=self.num_rotations + self.embedding_size)

        self.place_decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=5)
        self.place_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=self.num_rotations + self.embedding_size)

        self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], classes=5, dropout=0.5)

        self.embedding_head = nn.Sequential(
            nn.Linear(2*(self.embedding_size + 5), 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.offset = 0.0
    
    def forward(self, x):
        features = self.encoder(x)

        labels = self.classification_head(features[-1])

        output_fling = self.fling_head(self.fling_decoder(*features))
        output_fling_to_fold_sleeve = self.fling_to_fold_sleeve_head(self.fling_to_fold_sleeve_decoder(*features))
        output_fling_to_fold_bottom = self.fling_to_fold_bottom_head(self.fling_to_fold_bottom_decoder(*features))
        output_pick = self.pick_head(self.pick_decoder(*features))
        output_place = self.place_head(self.place_decoder(*features))

        return output_fling, output_fling_to_fold_sleeve, output_fling_to_fold_bottom, output_pick, output_place, labels

    def forward_combine(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.embedding_head(x)
        return x[:, 0]

    @staticmethod
    def expand_to(x, channels=20):
        return x.reshape(-1, 1, 1, 1).expand(-1, channels, 192, 256)

    @staticmethod
    def to_weight(heatmap, weight_range):
        weight = BiMamaNet.expand_to(weight_range[:, 0]) + heatmap * BiMamaNet.expand_to(weight_range[:, 1] - weight_range[:, 0])
        return weight / torch.mean(weight)

    @staticmethod
    def bce(prediction, target, weights=1.0):
        return (weights * nn.BCEWithLogitsLoss(reduction='none')(prediction, target.float())).sum(dim=(1, 2, 3)).mean()  # Sum over pixels, mean over batch

    @staticmethod
    def unravel_index(indices, shape): # from here: https://github.com/pytorch/pytorch/pull/66687
        shape = indices.new_tensor(shape + (1,))
        coefs = shape[1:].flipud().cumprod(dim=0).flipud()
        return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]

    def sample_point(self, heat, random=False):
        heat_flat = heat.flatten(start_dim=1) + 1e-18
        if random:
            heat_flat = self.sigmoid(heat_flat)
            heat_flat += self.offset
            heat_flat = torch.sqrt(heat_flat)

        dist = torch.distributions.Categorical(heat_flat)
        sample = dist.sample()
        return self.unravel_index(sample, heat[0].shape)

    def get_embedding(self, pred, ind):
        batch_range = torch.arange(0, len(pred))
        prob = pred[batch_range, ind[:, 0], ind[:, 1], ind[:, 2]].unsqueeze(-1)
        angle = 2 * np.pi * ind[:, 0].unsqueeze(-1) / 20
        x = 2 * ind[:, 1].unsqueeze(-1) / 192 - 1.0
        y = 2 * ind[:, 2].unsqueeze(-1) / 256 - 1.0
        embedding = pred[batch_range, 20:, ind[:, 1], ind[:, 2]]
        return torch.cat([prob, torch.cos(angle), torch.sin(angle), x, y, embedding], dim=1)

    def forward_loss(self, batch):
        image, fling1_heatmap, fling2_heatmap, fling_to_fold_sleeve_heatmap, fling_to_fold_bottom_heatmap, pick_heatmap, place_heatmap, reward, weight_range, primitive_index, annotation_weight = tuple(x.cuda() for x in batch)
        fling_heatmap = torch.stack([fling1_heatmap, fling2_heatmap]).mean(dim=0)

        pred_fling, pred_fling_to_fold_sleeve, pred_fling_to_fold_bottom, pred_pick, pred_place, pred_labels = self.forward(image)

        fling_weight = self.to_weight(fling_heatmap, weight_range[:, 0])
        fling_to_fold_sleeve_weight = self.to_weight(fling_to_fold_sleeve_heatmap, weight_range[:, 1])
        fling_to_fold_bottom_weight = self.to_weight(fling_to_fold_bottom_heatmap, weight_range[:, 1])
        pick_weight = self.to_weight(pick_heatmap, weight_range[:, 2])
        place_weight = self.to_weight(place_heatmap, weight_range[:, 2])

        # Embedding
        batch_range = torch.arange(0, len(primitive_index))
        zero_pred = torch.zeros_like(pred_fling)
        zero_heatmap = torch.zeros_like(fling1_heatmap)
        # primitive list: [fling, fling-to-fold, pick-and-hold, drag, done]
        pred1 = torch.stack([pred_fling, pred_fling_to_fold_sleeve, pred_pick, zero_pred, zero_pred], dim=1)[batch_range, primitive_index]
        pred2 = torch.stack([pred_fling, pred_fling_to_fold_bottom, pred_place, zero_pred, zero_pred], dim=1)[batch_range, primitive_index]
        heat1 = torch.stack([fling1_heatmap, fling_to_fold_sleeve_heatmap, pick_heatmap, zero_heatmap, zero_heatmap], dim=1)[batch_range, primitive_index]
        heat2 = torch.stack([fling2_heatmap, fling_to_fold_bottom_heatmap, place_heatmap, zero_heatmap, zero_heatmap], dim=1)[batch_range, primitive_index]

        h11, h12 = self.sample_point(heat1[:, :20]), self.sample_point(heat1[:, :20])
        h21, h22 = self.sample_point(heat2[:, :20]), self.sample_point(heat2[:, :20])
        p11, p12 = self.sample_point(pred1[:, :20], random=True), self.sample_point(pred1[:, :20], random=True)
        p21, p22 = self.sample_point(pred2[:, :20], random=True), self.sample_point(pred2[:, :20], random=True)

        emb1 = self.get_embedding(pred1, h11)
        emb2 = self.get_embedding(pred2, h21)
        emb3 = self.get_embedding(pred1, p11)
        emb4 = self.get_embedding(pred2, p21)
        emb5 = self.get_embedding(pred1, h12)
        emb6 = self.get_embedding(pred2, h22)
        emb7 = self.get_embedding(pred1, p12)
        emb8 = self.get_embedding(pred2, p22)

        x1 = torch.cat([emb1, emb5, emb1, emb3, emb7, emb3, emb7, emb5])
        x2 = torch.cat([emb2, emb6, emb4, emb2, emb6, emb4, emb8, emb1])
        
        pred_emb = self.forward_combine(x1, x2)
        pred_emb = pred_emb.reshape(8, -1).t()

        target_emb = torch.zeros_like(pred_emb)
        target_emb[:, 0] = reward
        target_emb[:, 1] = reward

        reward = self.expand_to(reward)

        pred_fling = pred_fling[:, :20]
        pred_fling_to_fold_sleeve = pred_fling_to_fold_sleeve[:, :20]
        pred_fling_to_fold_bottom = pred_fling_to_fold_bottom[:, :20]
        pred_pick = pred_pick[:, :20]
        pred_place = pred_place[:, :20]

        loss_fling = self.bce(pred_fling, reward * fling_heatmap, weights=fling_weight)
        loss_fling_to_fold_sleeve = self.bce(pred_fling_to_fold_sleeve, reward * fling_to_fold_sleeve_heatmap, weights=fling_to_fold_sleeve_weight)
        loss_fling_to_fold_bottom = self.bce(pred_fling_to_fold_bottom, reward * fling_to_fold_bottom_heatmap, weights=fling_to_fold_bottom_weight)
        loss_pick = self.bce(pred_pick, reward * pick_heatmap, weights=pick_weight)
        loss_place = self.bce(pred_place, reward * place_heatmap, weights=place_weight)

        loss_embedding = nn.BCEWithLogitsLoss(reduction='none')(pred_emb, target_emb.float())
        loss_embedding = 2500.0 * loss_embedding.mean()

        classification_weight = torch.tensor([1.0, 1.2, 2.0, 8.0, 16.0], device='cuda')
        loss_classification = annotation_weight * nn.CrossEntropyLoss(weight=classification_weight, reduction='none')(pred_labels, primitive_index)
        loss_classification = 80.0 * loss_classification.mean()

        mse_final_reward = nn.MSELoss()(self.sigmoid(pred_emb[:, 0]), target_emb[:, 0].float())

        pred_fling_to_fold = torch.cat((pred_fling_to_fold_sleeve, pred_fling_to_fold_bottom), dim=1)
        pred_pick_place = torch.cat((pred_pick, pred_place), dim=1)
        pred_primitive_index_reward = torch.argmax(torch.stack([
            pred_fling.amax(dim=(1, 2, 3)),
            pred_fling_to_fold.amax(dim=(1, 2, 3)),
            pred_pick_place.amax(dim=(1, 2, 3)),
        ]), dim=0)

        return {
            "loss": loss_fling + loss_fling_to_fold_sleeve + loss_fling_to_fold_bottom + loss_pick + loss_place + loss_embedding + loss_classification,
            "loss_fling": loss_fling,
            "loss_fling_to_fold": loss_fling_to_fold_sleeve + loss_fling_to_fold_bottom,
            "loss_picknhold": loss_pick + loss_place,
            "loss_embedding": loss_embedding,
            "loss_classification": loss_classification,
            "accuracy": (torch.argmax(pred_labels, dim=1) == primitive_index).float().mean(),
            "accuracy_reward": (pred_primitive_index_reward == primitive_index).float().mean(),
            "mse_final_reward": mse_final_reward,
        }
