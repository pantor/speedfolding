from time import time
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from selection import Max
from drawing import draw_pose_circle


class Prediction:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.model = self.model.cuda()

        self.angles = np.linspace(0.0, 2*np.pi, model.num_rotations, endpoint=False)
        self.angles[self.angles > np.pi] -= 2*np.pi

        self.transform_to_tensor = transforms.ToTensor()

        self.primitives = ['fling', 'fling-to-fold', 'pick-and-hold', 'drag', 'done']
        self.max_heatmap_without_embedding = False  # Just for ablative study

    @staticmethod
    def sample_embeddings(emb, heat, size):
        score = torch.square(heat)
        score_sum = torch.sum(score).item()
        score = score.detach().cpu().numpy()[0]
        emb = emb.detach().cpu().numpy()[0]

        f1 = np.random.choice(np.arange(score.size), size=size, replace=False, p=score.flatten() / score_sum)
        f1s = np.array([np.unravel_index(f, score.shape) for f in f1])

        emb_score1 = np.expand_dims(emb[f1s[:, 0], f1s[:, 1], f1s[:, 2]], -1)
        emb_angle1 = 2 * np.pi * np.expand_dims(f1s[:, 0], -1) / 20
        emb_x1 = 2 * np.expand_dims(f1s[:, 1], -1) / 192 - 1.0
        emb_y1 = 2 * np.expand_dims(f1s[:, 2], -1) / 256 - 1.0
        emb_emb1 = emb[20:, f1s[:, 1], f1s[:, 2]].transpose()

        return np.concatenate((emb_score1, np.cos(emb_angle1), np.sin(emb_angle1), emb_x1, emb_y1, emb_emb1), axis=1), f1s

    def find_two_local_maxima(self, selection, action_type: str, heat1, heat2):
        min_distance = 32 if action_type == 'fling' or action_type == 'fling-to-fold' or action_type == 'drag' else 4

        heat1 = heat1.detach().cpu().numpy()[0]
        heat2 = heat2.detach().cpu().numpy()[0]

        def action_iterator():
            for _ in range(heat1.size):
                idx1 = selection(heat1)
                h1, y1, x1 = np.unravel_index(idx1, heat1.shape)
                score1 = heat1[h1, y1, x1]

                for h in heat2:
                    cv.circle(h, (x1, y1), min_distance, (0, 0, 0), -1)

                idx2 = selection(heat2)
                h2, y2, x2 = np.unravel_index(idx2, heat2.shape)
                score2 = heat2[h2, y2, x2]

                poses = np.array([[x1, y1, self.angles[h1]], [x2, y2, self.angles[h2]]], dtype=np.float32) if action_type != 'done' else np.zeros((2, 3))
                scores = [score1, score2]

                yield action_type, poses, scores
                selection.disable(idx1, heat1)
                selection.disable(idx2, heat2)
            return None, None, None
        return action_iterator

    def predict(self, image, selection=None, action_type: str = None, left_mask=None, right_mask=None, return_timing=False):
        timing = {}
        start = time()

        selection = selection if selection else Max()

        image = self.transform_to_tensor(image).cuda()
        images = image.view(-1, image.shape[0], image.shape[1], image.shape[2])

        pre_processing = time()
        timing['pre_processing'] = pre_processing - start
        
        heatmaps_fling, heatmaps_fling_to_fold_sleeve, heatmaps_fling_to_fold_bottom, heatmaps_pick, heatmaps_place, label = self.model.forward(images)

        nn_inference = time()
        timing['nn_inference'] = nn_inference - pre_processing

        action = action_type if action_type else self.primitives[torch.argmax(label)]

        heatmaps = {
            'fling': torch.sigmoid(heatmaps_fling[:, :20]).detach().cpu().numpy()[0],
            'fling-to-fold-sleeve': torch.sigmoid(heatmaps_fling_to_fold_sleeve[:, :20]).detach().cpu().numpy()[0],
            'fling-to-fold-bottom': torch.sigmoid(heatmaps_fling_to_fold_bottom[:, :20]).detach().cpu().numpy()[0],
            'pick': torch.sigmoid(heatmaps_pick[:, :20]).detach().cpu().numpy()[0],
            'place': torch.sigmoid(heatmaps_place[:, :20]).detach().cpu().numpy()[0]
        }

        heat1 = heatmaps_fling[:, :20] if action == 'fling' else heatmaps_fling_to_fold_sleeve[:, :20] if action == 'fling-to-fold' else heatmaps_pick[:, :20]
        heat2 = heatmaps_fling[:, :20] if action == 'fling' else heatmaps_fling_to_fold_bottom[:, :20] if action == 'fling-to-fold' else heatmaps_place[:, :20]

        heat1 = torch.sigmoid(heat1)
        heat2 = torch.sigmoid(heat2)

        heat1_left = heat1.detach().clone()
        heat1_right = heat1.detach().clone()
        heat2_left = heat2.detach().clone()
        heat2_right = heat2.detach().clone()

        if left_mask is not None:
            left_mask = torch.from_numpy(left_mask.astype(np.float32)).cuda()
            heat1_left *= left_mask
            heat2_left *= left_mask

        if right_mask is not None:
            right_mask = torch.from_numpy(right_mask.astype(np.float32)).cuda()
            heat1_right *= right_mask
            heat2_right *= right_mask

        if self.max_heatmap_without_embedding:
            action_iterator = self.find_two_local_maxima(selection, action_type, heat1, heat2)
            if return_timing:
                return action_iterator, heatmaps, timing
            return action_iterator, heatmaps

        emb1 = heatmaps_fling if action == 'fling' else heatmaps_fling_to_fold_sleeve if action == 'fling-to-fold' else heatmaps_pick
        emb2 = heatmaps_fling if action == 'fling' else heatmaps_fling_to_fold_bottom if action == 'fling-to-fold' else heatmaps_place

        # Final size is 2 * size1 * size2 (so this is N/2 in comparison to the paper)
        size1 = 150
        size2 = 150

        x1_left, f1s_left = self.sample_embeddings(emb1, heat1_left, size1)
        x2_right, f2s_right = self.sample_embeddings(emb2, heat2_right, size2)

        x1_right, f1s_right = self.sample_embeddings(emb1, heat1_right, size1)
        x2_left, f2s_left = self.sample_embeddings(emb2, heat2_left, size2)

        sampling = time()
        timing['sampling'] = sampling - nn_inference

        x1_left = torch.from_numpy(x1_left.astype(np.float32)).cuda()
        x1_right = torch.from_numpy(x1_right.astype(np.float32)).cuda()
        x2_left = torch.from_numpy(x2_left.astype(np.float32)).cuda()
        x2_right = torch.from_numpy(x2_right.astype(np.float32)).cuda()

        x1_left = x1_left.repeat_interleave(size2, dim=0)
        x2_right = x2_right.repeat(size1, 1)

        x1_right = x1_right.repeat_interleave(size2, dim=0)
        x2_left = x2_left.repeat(size1, 1)

        x1 = torch.concat([x1_left, x1_right])
        x2 = torch.concat([x2_right, x2_left])

        final_reward = self.model.forward_combine(x1, x2)
        final_reward = torch.sigmoid(final_reward)
        final_reward = final_reward.detach().cpu().numpy()

        descriptor = time()
        timing['descriptor'] = descriptor - sampling

        def action_iterator():
            for _ in range(final_reward.size):
                final_idx = selection(final_reward)
                scores = final_reward[final_idx]

                number_combinations = size1 * size2
                if final_idx > number_combinations:
                    f1s, f2s = f1s_right, f2s_left
                    final_idx -= number_combinations
                else:
                    f1s, f2s = f1s_left, f2s_right
            
                final_idx1, final_idx2 = np.unravel_index(final_idx, (size1, size2))

                h1, y1, x1 = f1s[final_idx1]
                h2, y2, x2 = f2s[final_idx2]

                poses = np.array([[x1, y1, self.angles[h1]], [x2, y2, self.angles[h2]]], dtype=np.float32)
                if action == 'done' or action == 'drag':
                    poses = np.zeros((2, 3))
                    scores = torch.nn.Softmax(dim=1)(label)[0, 2].detach().cpu().numpy()

                yield action, poses, scores
                selection.disable(final_idx, final_reward)
            return None, None, None

        if return_timing:
            return action_iterator, heatmaps, timing

        return action_iterator, heatmaps
    
    def plot_single(self, img, heatmaps, image_id=0, pred_action=None, pred_keypoints=None, gt_keypoints=None):
        heatmaps['pick-and-hold'] = heatmaps['pick'] + heatmaps['place']
        heatmaps['fling-to-fold'] = heatmaps['fling-to-fold-sleeve'] + heatmaps['fling-to-fold-bottom']
        heatmaps['drag'] = heatmaps['fling-to-fold']

        heat = heatmaps[pred_action] if pred_action in heatmaps else np.zeros_like(heatmaps['fling'])
        heat = np.mean(heat, axis=0)

        heat = cv.normalize(heat, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        color_img = cv.cvtColor(img[:, :, 0], cv.COLOR_GRAY2RGB)
        vis = cv.applyColorMap(heat, cv.COLORMAP_JET)
                
        overlay = cv.addWeighted(color_img, 0.7, vis, 0.3, 0)
        if len(gt_keypoints) > 0:
            draw_pose_circle(overlay, gt_keypoints[0], (0, 255, 0))
            draw_pose_circle(overlay, gt_keypoints[1], (0, 255, 0))

        # for f in fs[1]:
        #     draw_pose_circle(overlay, f[::-1], (255, 150, 150))

        draw_pose_circle(overlay, pred_keypoints[0], (0, 0, 255))
        draw_pose_circle(overlay, pred_keypoints[1], (0, 0, 255), rect=True)
        overlay = cv.putText(overlay, pred_action, (15, 175), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.imwrite(f'preds/out{image_id:05d}.png', overlay)


if __name__ == '__main__':
    from dataset import BiFoldingDataset
    from model import BiMamaNet

    parser = ArgumentParser()
    parser.add_argument('model', metavar='model_path', type=Path, help='the path to the model')
    parser.add_argument('--collection', dest='collection', type=str, help='the database collection', default='test')
    parser.add_argument('-t', dest='type', type=str, help='type', default=None)

    args = parser.parse_args()

    model = BiMamaNet(num_rotations=20)
    model.load_state_dict(torch.load(args.model))
    prediction = Prediction(model)

    _, test_actions = BiFoldingDataset.load_train_test_actions(args.collection)
    test_dataset = BiFoldingDataset(args.collection, test_actions)

    fling_actions = list(filter(lambda a: a['type'] == 'fling' and a['is_human_annotated'], test_dataset.actions))
    pnh_actions = list(filter(lambda a: a['type'] == 'pick-and-hold' and a['is_human_annotated'], test_dataset.actions))
    done_actions = list(filter(lambda a: a['type'] == 'done' and a['is_human_annotated'], test_dataset.actions))

    actions = fling_actions[-30:-10] + pnh_actions[-7:] + done_actions[-3:]

    data_all = []
    for i, action in enumerate(actions):
        image = test_dataset.read_image(action['episode_id'], action['action_id'])
        poses = np.array([test_dataset.pose_to_array(p) for p in action['poses']])

        action_iterator, heatmaps = prediction.predict(image)
        for action_type, poses, scores in action_iterator():
            break
        data_all.append((i, image, poses, (action_type, poses, scores, heatmaps)))

    def plot(data):
        i, image, poses, (pred_action, pred_poses, pred_scores, heatmaps) = data
        print(i, pred_action, 'scores:', pred_scores)
        prediction.plot_single(image, heatmaps, image_id=i, pred_action=pred_action, pred_keypoints=pred_poses, gt_keypoints=poses)

    with Pool(10) as p:
        p.map(plot, data_all)
