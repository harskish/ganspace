swagger: '2.0'
info:
  title: Ganter API
  version: "0.1"
consumes:
  - application/json
produces:
  - application/json

basePath: /api

paths:
  /all_projects:
    get:
      tags:
        - all
      summary: information about all projects and sources available
      operationId: netdissect.server.get_all_projects
      responses:
        200:
          description: return list of projects
          schema:
            type: array
            items:
              type: object

  /layers:
    get:
      operationId: netdissect.server.get_layers
      tags:
        - all
      summary: returns information about all layers
      parameters:
        - $ref: '#/parameters/project'
      responses:
        200:
          description: Return requested data
          schema:
            type: object

  /units:
    get:
      operationId: netdissect.server.get_units
      tags:
        - all
      summary: returns unit information for one layer
      parameters:

        - $ref: '#/parameters/project'
        - $ref: '#/parameters/layer'

      responses:
        200:
          description: Return requested data
          schema:
            type: object

  /rankings:
    get:
      operationId: netdissect.server.get_rankings
      tags:
        - all
      summary: returns ranking information for one layer
      parameters:

        - $ref: '#/parameters/project'
        - $ref: '#/parameters/layer'

      responses:
        200:
          description: Return requested data
          schema:
            type: object

  /levels:
    get:
      operationId: netdissect.server.get_levels
      tags:
        - all
      summary: returns feature levels for one layer
      parameters:

        - $ref: '#/parameters/project'
        - $ref: '#/parameters/layer'
        - $ref: '#/parameters/quantiles'

      responses:
        200:
          description: Return requested data
          schema:
            type: object

  /features:
    post:
      summary: calculates max feature values within a set of image locations
      operationId: netdissect.server.post_features
      tags:
        - all
      parameters:
        - in: body
          name: feat_req
          description: RequestObject
          schema:
            $ref: "#/definitions/FeatureRequest"
      responses:
        200:
          description: returns feature vector for each layer

  /featuremaps:
    post:
      summary: calculates max feature values within a set of image locations
      operationId: netdissect.server.post_featuremaps
      tags:
        - all
      parameters:
        - in: body
          name: feat_req
          description: RequestObject
          schema:
            $ref: "#/definitions/FeatureMapRequest"
      responses:
        200:
          description: returns feature vector for each layer

  /channels:
    get:
      operationId: netdissect.server.get_channels
      tags:
        - all
      summary: returns channel information
      parameters:

        - $ref: '#/parameters/project'
        - $ref: '#/parameters/layer'

      responses:
        200:
          description: Return requested data
          schema:
            type: object

  /generate:
    post:
      summary: generates images for given zs constrained by ablation
      operationId: netdissect.server.post_generate
      tags:
        - all
      parameters:
        - in: body
          name: gen_req
          description: RequestObject
          schema:
            $ref: "#/definitions/GenerateRequest"
      responses:
        200:
          description: aaa


parameters:
  project:
    name: project
    description: project ID
    in: query
    required: true
    type: string

  layer:
    name: layer
    description: layer ID
    in: query
    type: string
    default: "1"

  quantiles:
     name: quantiles
     in: query
     type: array
     items:
       type: number
       format: float

definitions:

  GenerateRequest:
    type: object
    required:
      - project
    properties:
      project:
        type: string
      zs:
        type: array
        items:
          type: array
          items:
            type: number
            format: float
      ids:
        type: array
        items:
          type: integer
      return_urls:
        type: integer
      interventions:
        type: array
        items:
          - $ref: '#/definitions/Intervention'

  FeatureRequest:
    type: object
    required:
      - project
    properties:
      project:
        type: string
        example: 'churchoutdoor'
      layers:
        type: array
        items:
          type: string
        example: [ 'layer5' ]
      ids:
        type: array
        items:
          type: integer
      masks:
        type: array
        items:
          - $ref: '#/definitions/Mask'
      interventions:
        type: array
        items:
          - $ref: '#/definitions/Intervention'

  FeatureMapRequest:
    type: object
    required:
      - project
    properties:
      project:
        type: string
        example: 'churchoutdoor'
      layers:
        type: array
        items:
          type: string
        example: [ 'layer5' ]
      ids:
        type: array
        items:
          type: integer
      interventions:
        type: array
        items:
          - $ref: '#/definitions/Intervention'

  Intervention:
    type: object
    properties:
      maskalpha:
        $ref: '#/definitions/Mask'
      maskvalue:
        $ref: '#/definitions/Mask'
      ablations:
        type: array
        items:
          - $ref: '#/definitions/Ablation'

  Ablation:
    type: object
    properties:
      unit:
        type: integer
      alpha:
        type: number
        format: float
      value:
        type: number
        format: float
      layer:
        type: string

  Mask:
    type: object
    description: 2d bitmap mask
    properties:
      shape:
        type: array
        items:
          type: integer
        example: [ 128, 128 ]
      bitbounds:
        type: array
        items:
          type: integer
        example: [ 12, 42, 16, 46 ]
      bitstring:
        type: string
        example: '0110111111110011'

