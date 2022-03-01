from pathlib import Path

from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import torch

from dataset import BiFoldingDataset
from model import BiMamaNet
from prediction import Prediction


if __name__ == '__main__':
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
