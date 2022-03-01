from datetime import datetime
from hashlib import sha1
import json
import io
from pathlib import Path

import cv2 as cv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from database import Database
from drawing import draw_action, draw_masked_area


base_path = Path(__file__).parent.absolute().parent / 'data'

app = FastAPI()
db = {k: Database(k) for k in ['test']}


@app.get('/api/stats/{collection}')
async def get_stats(collection: str, needs_annotation: int = None, is_human_annotated: int = None, is_self_supervised: int = None, is_grasp_failure: int = None, primitive_type: str = None, reward: str = None):
    where_clause = db[collection].get_where_clause(needs_annotation, is_human_annotated, is_self_supervised, is_grasp_failure, primitive_type, reward)
    db[collection].c.execute(f"SELECT Count(*) as number_actions FROM actions {where_clause}")
    stats = dict(db[collection].c.fetchone())
    db[collection].c.execute(f"SELECT Count(*) as number_images FROM images LEFT JOIN actions ON images.episode_id == actions.episode_id AND images.action_id == actions.action_id {where_clause}")
    stats.update(dict(db[collection].c.fetchone()))
    return stats


@app.get('/api/actions/{collection}')
async def get_actions(collection: str, needs_annotation: int = None, is_human_annotated: int = None, is_self_supervised: int = None, is_grasp_failure: int = None, primitive_type: str = None, reward: str = None, limit: int = None, skip: int = None):
    return list(a for a in db[collection].yield_actions(needs_annotation, is_human_annotated, is_self_supervised, is_grasp_failure, primitive_type, reward, limit, skip))


@app.get('/api/actions/index/{collection}/{episode_id}/{action_id}')
async def get_action_index(collection: str, episode_id: str, action_id: int, needs_annotation: int = None, is_human_annotated: int = None, is_self_supervised: int = None, is_grasp_failure: int = None, primitive_type: str = None, reward: str = None):
    where_clause = db[collection].get_where_clause(needs_annotation, is_human_annotated, is_self_supervised, is_grasp_failure, primitive_type, reward) 

    db[collection].c.execute(f"SELECT episode_id, action_id, data FROM actions {where_clause} ORDER BY episode_id DESC, action_id ASC")
    for i, row in enumerate(db[collection].c.fetchall()):
        if row['episode_id'] <= episode_id and row['action_id'] == action_id:
            return {'index': i}
    return {'index': 0}


@app.get('/api/action/{collection}/{episode_id}/{action_id}')
async def get_action(collection: str, episode_id: str, action_id: int):
    return db[collection].get_action(episode_id, action_id)


@app.get('/api/scene/{collection}/{episode_id}/{action_id}/{scene}/equal')
async def get_equal_scenes(collection: str, episode_id: str, action_id: str, scene: str):
    db[collection].c.execute(f"SELECT actions.id as 'row_id', images.episode_id, images.action_id, images.scene, images.hash, json_extract(actions.data, '$.type') as 'type' FROM images JOIN actions ON images.episode_id == actions.episode_id AND images.action_id == actions.action_id WHERE hash == (SELECT hash FROM images WHERE episode_id == ? AND action_id == ? AND scene == ? AND camera = 'color') AND images.episode_id != ?", (episode_id, action_id, scene, episode_id))
    rows = db[collection].c.fetchall()
    return {'equal_scenes': list(rows)}


def send_image(image=None, webp=False):
    if image is None:
        empty = np.zeros((772, 1032, 1))
        cv.putText(empty, '?', (386 + 70, 516 - 76), cv.FONT_HERSHEY_SIMPLEX, 6, 100, thickness=6)
        return send_image(empty)

    if webp:
        encode_param = [int(cv.IMWRITE_WEBP_QUALITY), 85]
        _, image_encoded = cv.imencode('.webp', image, encode_param)
        return StreamingResponse(io.BytesIO(image_encoded), media_type='image/webp')

    _, image_encoded = cv.imencode('.png', image)
    return StreamingResponse(io.BytesIO(image_encoded), media_type='image/png')

@app.get('/api/image/current/{camera}')
async def get_image(camera: str):
    camera = f'image-{camera}' if camera in ['color', 'depth'] else camera
    image = cv.imread(str(base_path / 'current' / f'{camera}.png'))
    return send_image(image)


@app.get('/api/image/{collection}/{episode_id}/{action_id}/{scene}/{camera}')
async def get_image(collection: str, episode_id: str, action_id: str, scene: str, camera: str, draw_pose: bool = False, draw_mask: bool = True):
    image = db[collection].get_image(episode_id, action_id, scene, camera)
    if image is None:
        return send_image(image)

    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    if draw_pose:
        action = await get_action(collection, episode_id, action_id)
        draw_action(image, action)

    if draw_mask:
        draw_masked_area(
            image,
            mask_nn_input={'top': 0, 'right': 12, 'bottom': 116, 'left': 145},
            mask_heatmap={'top': 45, 'right': 6, 'bottom': 16, 'left': 30},
        )

    return send_image(image)


@app.post('/api/image')
async def upload_image(collection: str = Form(...), episode_id: str = Form(...), action_id: int = Form(...), scene: str = Form(...), camera: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    image_buffer = np.frombuffer(image_data, np.uint8)
    image = cv.imdecode(image_buffer, cv.IMREAD_UNCHANGED)

    if episode_id == 'current':
        image_path = base_path / 'current' / f'image-{camera}.png'
        image_path.parent.mkdir(exist_ok=True, parents=True)
        cv.imwrite(str(image_path), image)
        return True

    image_path = base_path / collection / episode_id / f'{action_id}-{scene}-{camera}.png'
    image_path.parent.mkdir(exist_ok=True, parents=True)

    cv.imwrite(str(image_path), image)

    h = sha1(image)
    db[collection].c.execute("INSERT INTO images (episode_id, action_id, scene, camera, hash) VALUES (?, ?, ?, ?, ?)", (episode_id, action_id, scene, camera, h.hexdigest()))
    db[collection].conn.commit()
    return (db[collection].c.rowcount == 1)


class NewAction(BaseModel):
    collection: str
    episode_id: str
    action_id: int
    data: str

@app.post('/api/action')
async def upload_action(action: NewAction):
    db[action.collection].c.execute("INSERT INTO actions (episode_id, action_id, data) VALUES (?, ?, json(?))", (action.episode_id, action.action_id, action.data))
    db[action.collection].conn.commit()
    return (db[action.collection].c.rowcount == 1)


class NewAnnotation(BaseModel):
    primitive_type: str
    poses: str # json

@app.post('/api/action/{collection}/{episode_id}/{action_id}/annotation')
async def upload_annotation(collection: str, episode_id: str, action_id: int, annotation: NewAnnotation):
    annotation_done = (annotation.poses != '[]' or annotation.primitive_type in ('done', 'drag'))

    db[collection].c.execute("UPDATE actions SET data = json_set(data, '$.type', ?) WHERE episode_id = ? and action_id = ?", (annotation.primitive_type, episode_id, action_id));
    db[collection].c.execute("UPDATE actions SET data = json_set(data, '$.poses', json(?)) WHERE episode_id = ? and action_id = ?", (annotation.poses, episode_id, action_id));
    db[collection].c.execute(f"UPDATE actions SET data = json_set(data, '$.needs_annotation', {0 if annotation_done else 1}) WHERE episode_id = ? and action_id = ?", (episode_id, action_id));
    db[collection].c.execute(f"UPDATE actions SET data = json_set(data, '$.is_human_annotated', {1 if annotation_done else 0}) WHERE episode_id = ? and action_id = ?", (episode_id, action_id));
    db[collection].conn.commit()
    return {'success': (db[collection].c.rowcount == 1)}


@app.post('/api/episode/{collection}/{episode_id}/{action_id}/{scene}/copy')
async def copy_episode(collection: str, episode_id: str, action_id: int, scene: str):
    new_episode_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
    new_scene = 'before' if scene == 'after' else scene
    (base_path / collection / new_episode_id).mkdir()

    # Copy images
    db[collection].c.execute("SELECT * FROM images WHERE episode_id == ? AND action_id == ? AND scene == ?", (episode_id, action_id, scene))
    for row in db[collection].c.fetchall():
        image = cv.imread(str(base_path / collection / row['episode_id'] / f"{row['action_id']}-{row['scene']}-{row['camera']}.png"))
        cv.imwrite(str(base_path / collection / new_episode_id / f"{row['action_id']}-{new_scene}-{row['camera']}.png"), image)
        db[collection].c.execute("INSERT INTO images (episode_id, action_id, scene, camera, hash) VALUES (?, ?, ?, ?, ?)", (new_episode_id, row['action_id'], new_scene, row['camera'], row['hash']))

    # Copy action
    db[collection].c.execute("SELECT episode_id, action_id, data FROM actions WHERE episode_id = ? AND action_id = ?", (episode_id, action_id))
    action = dict(db[collection].c.fetchone())
    data = json.loads(action['data'])
    data['is_human_annotated'] = 1
    data['needs_annotation'] = 1
    data['is_self_supervised'] = 0
    data['copied_from'] = [{'episode_id': episode_id, 'action_id': action_id, 'scene': scene}]
    if 'reward' in data:
        del data['reward']
    db[collection].c.execute("INSERT INTO actions (episode_id, action_id, data) VALUES (?, ?, json(?))", (new_episode_id, action['action_id'], json.dumps(data)))

    db[collection].conn.commit()
    return {'success': (db[collection].c.rowcount == 1), 'episode_id': new_episode_id, 'action_id': action_id}


@app.post('/api/episode/{collection}/{episode_id}/delete')
async def delete_episode(collection: str, episode_id: str):
    db[collection].c.execute("DELETE FROM images WHERE episode_id == ?", (episode_id,))
    db[collection].c.execute("DELETE FROM actions WHERE episode_id == ?", (episode_id,))
    db[collection].conn.commit()
    return {'success': (db[collection].c.rowcount >= 1)}
