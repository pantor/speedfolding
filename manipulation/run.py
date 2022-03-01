from argparse import ArgumentParser
from time import sleep
from pathlib import Path
import pickle
import numpy as np

from loguru import logger

from evaluation import Evaluation
from experiment import Experiment
from inference import Inference
from reward import segment, calculate_reward
from saver import Episode
from selection import Max, Top


parser = ArgumentParser()
parser.add_argument('-n', '--number', dest='num_trials', type=int, default=int(1e6), help='number of experiment trials')
parser.add_argument('--horizon', dest='horizon', type=int, default=3, help='max number of steps before declaring failure')
parser.add_argument('--ablative', dest='ablative', type=str, default=None, help='some ablative parameters, e.g. flingbot, max_heatmap_without_embedding')
parser.add_argument('--save', action='store_true', help='saves the actions in the database for self-supervised learning')
parser.add_argument('--demo', action='store_true', help='this is for demos and videos')
parser.add_argument('--fold', dest='fold', type=str, default=None, help='name of the folding instruction, e.g. shirt or towel')
parser.add_argument('--reset-at-startup', action='store_true', help='reset the environment before doing anything else (useful for run_forever)')
parser.add_argument('--eval-save', action='store_true', help='saves the state of the evaluation to data/current/eval.pkl')
parser.add_argument('--eval-load', action='store_true', help='loads the state of the evaluation from data/current/eval.pkl')
parser.add_argument('--exit-on-grasp-failure', action='store_true', help='exits (and then ignores) when a grasp failure occured')
parser.add_argument('--fling-to-fold', action='store_true', help='fold the shirt directly from flinging')
parser.add_argument('--coverage', action='store_true', help='calculates change in coverage after each action')


def on_exit():
    evaluation.add_event('exit')
    
    if args.eval_save:
        logger.info(evaluation)
        pickle.dump(evaluation, open(evaluation_path, 'wb'))


if __name__ == '__main__':
    args = parser.parse_args()

    selection = Max() if args.demo else Top(20)

    exp = Experiment(on_exit_callback=on_exit)
    exp.home()

    inference = Inference(
        multi_model_name='multi-22022022-emb-f2f.pth',
        primitives_model_name='weights_1111_2022-02-04.pth',
        experiment=exp,
    )

    if args.ablative == 'flingbot':
        del inference

        logger.info('Test FlingBot Inference')
        from flingbot.inference import FlingBotInference
        inference = FlingBotInference(model_name='flingbot.pth')

    if args.ablative == 'max-heatmap-without-embedding':
        inference.prediction.max_heatmap_without_embedding = True  # Uncomment for ablative study without embeddings

    if args.reset_at_startup:
        image_before_normal, _ = exp.take_image()
        exp.gen_random_scene(image_before_normal)
        exp.home()

    evaluation_path = Path.home() / 'data' / 'current' / 'evaluation-state.pkl'
    evaluation = pickle.load(open(evaluation_path, 'rb')) if args.eval_load else Evaluation(horizon=args.horizon)
    
    for tidx in range(args.num_trials):
        evaluation.add_event('episode')
        if args.coverage:
            exp.hands_up()
            coverage = exp.take_coverage()
            logger.info(f"Current coverage is {coverage:0.3f}")
            exp.home()
            evaluation.episodes[-1][-1]['coverage'] = coverage

        with evaluation.profile('camera'):
            image_before_normal, image_before_ortho = exp.take_image()
            mask_before, info_before = segment(image_before_normal.color.data)
        
        for steps in range(args.horizon + 1):
            episode = Episode()
            
            with evaluation.profile('predict'):
                action = inference.predict_action(image_before_normal, selection, save=True)

                if action['type'] == 'f2f-out-of-reach':
                    action = inference.predict_action(image_before_normal, selection, action_type='fling', save=True)
            
            if action['type'] == 'done':
                evaluation.add_event(action['type'])
                if args.coverage:
                    evaluation.episodes[-1][-1]['coverage'] = coverage

                logger.info(f"Done with probability {action['score']:0.3f}, area of shirt is {info_before['area']:0.3f}")
                logger.info(f'Trial {tidx} is done after executing {steps} actions executed')

                if args.save:
                    logger.info(f'Saving data collected on trial {tidx} step {steps + 1}')
                
                    exp.saver.save_image(image_before_normal, 'test', episode.id, len(episode.actions), scene='before')
                    exp.saver.save_image(image_before_ortho, 'test', episode.id, len(episode.actions), scene='before', camera='ortho')
                    exp.saver.save_action(episode.id, len(episode.actions), data={'needs_annotation': True, 'is_human_annotated': False, 'is_self_supervised': True, 'type': action['type'], 'poses': action['poses'], 'reward': float(action['score'])})

                if args.fold:
                    logger.info(f'Start folding now...')
                    instruction = exp.tm.get_matched_instruction(mask_before, template_name=args.fold)

                    if args.fold == '2s':
                        exp.execute_2s_fold(image_before_normal, evaluation)
                    else:
                        exp.execute_fold(instruction, image_before_normal, evaluation, final_fling=False)
                    evaluation.add_event('folded')

                sleep(3.0) # [s] Wait to enjoy and admire the done state
                break

            if steps == args.horizon:
                break

            logger.info(f"Inferred action of type {action['type']} with score {action['score']:0.5f}")

            if action['score'] < 0.0 and args.ablative != 'flingbot':  # 0.01
                logger.info(f'Score is below threshold')
                break

            if action['type'] == 'fling-to-fold':
                if exp.mh.should_move_for_folding(image_before_normal.color.raw_data):
                    with evaluation.profile(f"move-smooth"):
                        exp.move_to_center(image=image_before_normal, x_only=True)
                        exp.home()

                    with evaluation.profile('camera'):
                        image_before_normal, _ = exp.take_image()

                    with evaluation.profile('predict'):
                        action = inference.predict_action(image_before_normal, selection, save=True)

            with evaluation.profile(action['type']):
                exp.execute_action(action, image_before_normal)
                    
                if args.coverage:
                    exp.hands_up()
                    coverage = exp.take_coverage()
                    logger.info(f"Current coverage is {coverage:0.3f}")
                    exp.home()

                    evaluation.add_event(action['type'], coverage=coverage)

            if action['type'] == 'fling-to-fold':
                exp.home()
                image_before_normal, image_before_ortho = exp.take_image()
                
                exp.execute_fold_after_f2f(inference, image_before_normal, evaluation)
                evaluation.add_event('folded')
                sleep(4.0)
                break

            with evaluation.profile('camera'):
                image_after_normal, image_after_ortho = exp.take_image()
                mask_after, info_after = segment(image_after_normal.color.data)
                
                if not args.demo:
                    ys, xs = np.where(mask_after > 0)
                    middle = mask_after.shape[1] / 2
                    if (min(xs) > middle or max(xs) < middle) and (max(ys) - min(ys) > 0.66 * mask_after.shape[0]):
                        logger.info('Grasp failure.')
                        evaluation.add_event('grasp-failure')
                        evaluation.add_event('grasp')
                        if args.exit_on_grasp_failure:
                            evaluation.add_event('exit')
                            break 
                    
                    else:  # Grasp success
                        evaluation.add_event('grasp')
                        evaluation.add_event('grasp')
            
            if args.save:
                reward = calculate_reward(mask_before, mask_after)
                logger.info(f'Reward on trial {tidx} and step {steps + 1}: {reward:0.5f}')

                logger.info(f'Saving data collected on trial {tidx} step {steps + 1}')
                
                exp.saver.save_image(image_before_normal, 'test', episode.id, len(episode.actions), scene='before')
                exp.saver.save_image(image_before_ortho, 'test', episode.id, len(episode.actions), scene='before', camera='ortho')
                exp.saver.save_action(episode.id, len(episode.actions), data={'needs_annotation': False, 'is_human_annotated': False, 'is_self_supervised': True, 'type': action['type'], 'poses': action['poses'], 'reward': reward})
                exp.saver.save_image(image_after_normal, 'test', episode.id, len(episode.actions), scene='after')
                exp.saver.save_image(image_after_ortho, 'test', episode.id, len(episode.actions), scene='after', camera='ortho')
            
            image_before_normal, image_before_ortho = image_after_normal, image_after_ortho
            mask_before, info_before = mask_after, info_after
            
            if steps == args.horizon - 1:
                logger.info(f"Failed to smooth the t-shirt. Starting trial {tidx + 1}")

        if args.eval_save:
            pickle.dump(evaluation, open(evaluation_path, 'wb'))
        logger.info(evaluation)

        # Reset to start a new trial
        with evaluation.profile('reset'):
            exp.gen_random_scene(image_before_normal)
            exp.home()

    logger.info("Actually scratch that... I'm done for today. See ya!")
