#this script generates gt bounding box labels for existing CLEVR dataset.

from __future__ import print_function
import argparse, json, os, itertools, random, shutil
import time
import re
import utils

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--input_scene_file', default='../output/CLEVR_scenes.json',
    help="JSON file containing ground-truth scene information for all images " +
         "from render_images.py")
parser.add_argument('--metadata_file', default='metadata.json',
    help="JSON file containing metadata about functions")
parser.add_argument('--synonyms_json', default='synonyms.json',
    help="JSON file defining synonyms for parameter values")
parser.add_argument('--template_dir', default='CLEVR_1.0_templates',
    help="Directory containing JSON templates for questions")

parser.add_argument('--output_questions_file',
    default='../output/CLEVR_questions.json',
    help="The output file to write containing generated questions")

# Control which and how many images to process
parser.add_argument('--scene_start_idx', default=0, type=int,
    help="The image at which to start generating questions; this allows " +
         "question generation to be split across many workers")
parser.add_argument('--num_scenes', default=0, type=int,
    help="The number of images for which to generate questions. Setting to 0 " +
         "generates questions for all scenes in the input file starting from " +
         "--scene_start_idx")


from copy import deepcopy
def get_grounded_obj(q, scene):
  q_graph = q
  vertex_idx = len(q_graph) - 1  # last item's idx
  from collections import deque
  queue = deque([q_graph[vertex_idx]])
  visited = {vertex_idx: True}
  grounded = []
  while queue:
    v = queue.popleft()
    if type(v['_output']) != list:
      neighbors = v['inputs']
      for n in neighbors:
        if n not in visited:
          queue.append(q_graph[n])
          visited[n] = True
    else:
      grounded.append(v['_output'])

  obj_idx = [j for i in grounded for j in i]
  objects = [scene['objects'][i] for i in obj_idx]
  d = dict(zip(obj_idx, objects))

  return [d, grounded]


if __name__ == '__main__':
    args = parser.parse_args()
    questions = utils.load_json('/media/data/CLEVR_v1.0/questions/CLEVR_val_questions_old.json')
    # with open(args.metadata_file, 'r') as f:
    #     metadata = json.load(f)
    #     dataset = metadata['dataset']
    #     if dataset != 'CLEVR-v1.0':
    #         raise ValueError('Unrecognized dataset "%s"' % dataset)

    # functions_by_name = {}
    # for f in metadata['functions']:
    #     functions_by_name[f['name']] = f
    # metadata['_functions_by_name'] = functions_by_name

    #read files containing questions


    # Read file containing input scenes
    all_scenes = []
    with open(args.input_scene_file, 'r') as f:
        scene_data = json.load(f)
        all_scenes = scene_data['scenes']
        scene_info = scene_data['info']
    begin = args.scene_start_idx
    if args.num_scenes > 0:
        end = args.scene_start_idx + args.num_scenes
        all_scenes = all_scenes[begin:end]
    else:
        all_scenes = all_scenes[begin:]

    # questions = []
    scene_count = 0
    img_id2scene_dict = {k['image_index']: k for k in all_scenes}
    for q in questions['questions']:
        scene = img_id2scene_dict[q['image_index']]
        grounded = get_grounded_obj(q['program'], scene)
        q['grounded_objects']: grounded

    utils.save_json(questions, args.output_questions_file)
    # for i, scene in enumerate(all_scenes):
    #     scene_fn = scene['image_filename']
    #     scene_struct = scene
    #     print('starting image %s (%d / %d)'
    #           % (scene_fn, i + 1, len(all_scenes)))
    #
    #     if scene_count % args.reset_counts_every == 0:
    #         print('resetting counts')
    #         template_counts, template_answer_counts = reset_counts()
    #     scene_count += 1

