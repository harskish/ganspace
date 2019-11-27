#!/usr/bin/env python

import argparse, connexion, os, sys, yaml, json, socket
from netdissect.easydict import EasyDict
from flask import send_from_directory, redirect
from flask_cors import CORS


from netdissect.serverstate import DissectionProject

__author__ = 'Hendrik Strobelt, David Bau'

CONFIG_FILE_NAME = 'dissect.json'
projects = {}

app = connexion.App(__name__, debug=False)


def get_all_projects():
    res = []
    for key, project in projects.items():
        # print key
        res.append({
            'project': key,
            'info': {
              'layers': [layer['layer'] for layer in project.get_layers()]
            }
        })
    return sorted(res, key=lambda x: x['project'])

def get_layers(project):
    return {
        'request': {'project': project},
        'res': projects[project].get_layers()
    }

def get_units(project, layer):
    return {
        'request': {'project': project, 'layer': layer},
        'res': projects[project].get_units(layer)
    }

def get_rankings(project, layer):
    return {
        'request': {'project': project, 'layer': layer},
        'res': projects[project].get_rankings(layer)
    }

def get_levels(project, layer, quantiles):
    return {
        'request': {'project': project, 'layer': layer, 'quantiles': quantiles},
        'res': projects[project].get_levels(layer, quantiles)
    }

def get_channels(project, layer):
    answer = dict(channels=projects[project].get_channels(layer))
    return {
        'request': {'project': project, 'layer': layer},
        'res': answer
    }

def post_generate(gen_req):
    project = gen_req['project']
    zs = gen_req.get('zs', None)
    ids = gen_req.get('ids', None)
    return_urls = gen_req.get('return_urls', False)
    assert (zs is None) != (ids is None) # one or the other, not both
    ablations = gen_req.get('ablations', [])
    interventions = gen_req.get('interventions', None)
    # no z avilable if ablations
    generated = projects[project].generate_images(zs, ids, interventions,
            return_urls=return_urls)
    return {
        'request': gen_req,
        'res': generated
    }

def post_features(feat_req):
    project = feat_req['project']
    ids = feat_req['ids']
    masks = feat_req.get('masks', None)
    layers = feat_req.get('layers', None)
    interventions = feat_req.get('interventions', None)
    features = projects[project].get_features(
            ids, masks, layers, interventions)
    return {
        'request': feat_req,
        'res': features
    }

def post_featuremaps(feat_req):
    project = feat_req['project']
    ids = feat_req['ids']
    layers = feat_req.get('layers', None)
    interventions = feat_req.get('interventions', None)
    featuremaps = projects[project].get_featuremaps(
            ids, layers, interventions)
    return {
        'request': feat_req,
        'res': featuremaps
    }

@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``

    :param path: path from api call
    """
    return send_from_directory(args.client, path)

@app.route('/data/<path:path>')
def send_data(path):
    """ serves all files from the data dir to ``/dissect/<path:path>``

    :param path: path from api call
    """
    print('Got the data route for', path)
    return send_from_directory(args.data, path)


@app.route('/')
def redirect_home():
    return redirect('/client/index.html', code=302)


def load_projects(directory):
    """
    searches for CONFIG_FILE_NAME in all subdirectories of directory
    and creates data handlers for all of them

    :param directory: scan directory
    :return: null
    """
    project_dirs = []
    # Don't search more than 2 dirs deep.
    search_depth = 2 + directory.count(os.path.sep)
    for root, dirs, files in os.walk(directory):
        if CONFIG_FILE_NAME in files:
            project_dirs.append(root)
            # Don't get subprojects under a project dir.
            del dirs[:]
        elif root.count(os.path.sep) >= search_depth:
            del dirs[:]
    for p_dir in project_dirs:
        print('Loading %s' % os.path.join(p_dir, CONFIG_FILE_NAME))
        with open(os.path.join(p_dir, CONFIG_FILE_NAME), 'r') as jf:
            config = EasyDict(json.load(jf))
            dh_id = os.path.split(p_dir)[1]
            projects[dh_id] = DissectionProject(
                    config=config,
                    project_dir=p_dir,
                    path_url='data/' + os.path.relpath(p_dir, directory),
                    public_host=args.public_host)

app.add_api('server.yaml')

# add CORS support
CORS(app.app, headers='Content-Type')

parser = argparse.ArgumentParser()
parser.add_argument("--nodebug", default=False)
parser.add_argument("--address", default="127.0.0.1") # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--public_host", default=None)
parser.add_argument("--nocache", default=False)
parser.add_argument("--data", type=str, default='dissect')
parser.add_argument("--client", type=str, default='client_dist')

if __name__ == '__main__':
    args = parser.parse_args()
    for d in [args.data, args.client]:
        if not os.path.isdir(d):
            print('No directory %s' % d)
            sys.exit(1)
    args.data = os.path.abspath(args.data)
    args.client = os.path.abspath(args.client)
    if args.public_host is None:
        args.public_host = '%s:%d' % (socket.getfqdn(), int(args.port))
    app.run(port=int(args.port), debug=not args.nodebug, host=args.address,
            use_reloader=False)
else:
    args, _ = parser.parse_known_args()
    if args.public_host is None:
        args.public_host = '%s:%d' % (socket.getfqdn(), int(args.port))
    load_projects(args.data)
