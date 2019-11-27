<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.3.1.js" integrity="sha256-2Kok7MbOyxpgUVvAk/HJ2jigOSYS2auK4Pfzbm7uH60=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vue/2.5.16/vue.js" integrity="sha256-CMMTrj5gGwOAXBeFi7kNokqowkzbeL8ydAJy39ewjkQ=" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/lodash@4.17.10/lodash.js" integrity="sha256-qwbDmNVLiCqkqRBpF46q5bjYH11j5cd+K+Y6D3/ja28=" crossorigin="anonymous"></script>
<script src="https://unpkg.com/split.js@1.3.5/split.js" integrity="sha384-zUUFqW6+QulITI/GAXB5UnHBryKHEZp4G+eNNC1/3Z3IJ+6CZRSFcdl4f7gCa957" crossorigin="anonymous"></script>
<style>
[v-cloak] {
  display: none;
}
body, html { height: 100vh; }
#app {
  border: 0;
  margin: 0;
  padding: 0;
  height: 100vh;
  width: 100vw;
}
table.pane {
  float: left;
  display: inline-table;
  border: 0;
  margin: 0;
  padding: 0;
  border-spacing: 0;
  height: 100vh;
  width: 50vw;
}
.gutter-horizontal {
  float: left;
  height: 100vh;
  background-color: #eee;
  cursor: ew-resize;
}
.palette, .preview {
  height: 100%;
  overflow-y: scroll;
  text-align: center;
}
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
  max-height: 100%;
}
.unitgrid .info {
  text-align: left;
  display: block;
  position: absolute;
  top: 0;
  color: white;
}
.unitgrid .layername {
}
.unitgrid .unitnum {
}
.unitlabel {
  font-weight: bold;
  line-height: 1;
  position: absolute;
  bottom: 0;
  color: white;
  white-space: nowrap;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 72px;
  height: 72px;
}
.thumbcrop img, .img-scroller img {
  image-rendering: pixelated;
}
.unit {
  display: inline-block;
  background: white;
  padding: 0;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
  overflow: hidden;
  height: 72px;
  width: 72px;
  position: relative;
  user-select: none;
}
.unit.ablated .ablationmark::after,
.selmodeablation .unit.dragged .ablationmark::after {
  content: '\2716';
  bottom: 0;
}
.unit.inserted .insertionmark::after,
.selmodeinsertion .unit.dragged .insertionmark::after {
  content: '\2713';
  font-weight: bold;
  text-shadow: 0 0 #000;
  bottom: 10px;
}
.ablationmark::after, .insertionmark::after {
  line-height: 1;
  position: absolute;
  right: 0;
}
.unit.ablated .ablationmark {
  color: red;
}
.unit.inserted .insertionmark {
  color: lime;
}
.unit.dragged {
  opacity: 0.8;
}
.selmodeablation .unit.dragged .ablationmark,
.selmodeinsertion .unit.dragged .insertionmark {
  color: yellow;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}
.bg-inverse {
  background: #021B54;
}
.layout {
  width: 50%;
  vertical-align: top;
}
.chooser-cell {
  height: 100px;
  background: #021B54;
  color: white;
  padding: 5px 5px 0;
}
.examplepair {
  display: inline-block;
  margin: 2px;
  position: relative;
  line-height: 0;
}
.exampleid {
  position: absolute;
  line-height: 1;
  font-size: large;
  font-weight: bold;
  text-shadow: -1px 2px #000, 1px 2px #000;
  bottom: 0;
  color: white;
  z-index: 1;
}
.querymark, .editmark {
  position: absolute;
  background: rgba(255,0,0,0.5);
  width: 10px;
  height: 10px;
  margin: -5px;
  border-radius: 5px;
  top: 0;
  left: 0;
  pointer-events: none;
}
.editmark {
  background: rgba(0,255,0,0.5);
}
.chooser-cell input[type=button] {
  border-radius: 5px; padding: 0 5px;
}
</style>
</head>
<body class="unitviz">
<div id="app" :class="['selmode' + selection_mode]" v-cloak>
<table class="pane" id="leftpane">
<tr>
<td class="chooser-cell">
<div v-if="dissect">
<p>
<label for="selmode-ablation-radio">
<input type="radio" id="selmode-ablation-radio"
       value="ablation" v-model="selection_mode">
<span style="color:red">&#x2716;</span>
Choose units to ablate.</label>
<label for="selmode-insertion-radio">
<input type="radio" id="selmode-insertion-radio"
       value="insertion" v-model="selection_mode">
<span style="color:lime;font-weight:bold">&#x2713;</span>
Choose units to insert.</label>
</p>
<label for="ranking-radio">
<input type="radio" id="ranking-radio" value="ranking" v-model="palette_mode">
Show all units in
<select v-model="palette_layer" v-on:change="palette_mode='ranking'">
<option v-for="(lrec, lind) in dissect.layers" :value="lind"
>{{lrec.layer}}</option>
</select>,
sorted by
<template v-for="lrec in [dissect.layers[palette_layer]]">
<select v-model="sort_order" v-on:change="palette_mode='ranking'">
<option v-for="rank in lrec['rankings']" :value="rank.name"
>{{rank.name}}</option>
</select>
</template>
</label><br>
<label for="ablation-radio">
<input type="radio" id="ablation-radio" value="ablation" v-model="palette_mode">
Show current ablation
        ({{ _.sum(_.map(selected_ablation,
              function(x) { return _.sum(x)} ))
          }} units ablated)</label>
<input type="button" value="Reset" v-on:click="resetselection('ablation')">
<input type="button" :value="'Invert ' + dissect.layers[palette_layer].layer"
       v-on:click="invertselection('ablation')"><br>
<label for="insertion-radio">
<input type="radio" id="insertion-radio" value="insertion"
       v-model="palette_mode">
Show current insertion
        ({{ _.sum(_.map(selected_insertion,
              function(x) { return _.sum(x)} ))
          }} units inserted)</label>
<input type="button" value="Reset" v-on:click="resetselection('insertion')">
<input type="button" :value="'Invert ' + dissect.layers[palette_layer].layer"
       v-on:click="invertselection('insertion')"><br>
<label for="query-radio">
<input type="radio" id="query-radio" value="query" v-model="palette_mode"
 :disabled="!(dissect.layers[palette_layer].layer in query_ranking)">
Units in {{ dissect.layers[palette_layer].layer }}
by 
<select v-model="query_stat">
<option value="mean_quantile">quantile of average</option>
<option value="max_quantile">quantile of maximum</option>
<option value="mean">average activation</option>
<option value="max">maximum activation</option>
</select>
activation on <span style="color:red">&#x25cf;</span> selected pixels of
{{ currentquery.map(x => 'image #' + x.id).join(', ') || 'an image' }}.
</label>
</div>
</td>
</tr>

<tr>
<td class="layout">

<div class="palette" v-if="dissect">

<div class="unitgrid"
><div :class="{unit: true, lowscore: !urec.interp,
               ablated: (selected_ablation[urec.layer][urec.unit] > 0),
               inserted: (selected_insertion[urec.layer][urec.unit] > 0),
               dragged: (dragging.active &&
                ((dragging.first <= ordernum && ordernum <= dragging.last) ||
                 (dragging.last <= ordernum && ordernum <= dragging.first)))
               }"
      v-for="urec, ordernum in palette_units"
      :data-ordernum="ordernum"
      >
<template v-if="sort_order.indexOf('-') < 0"
          ><div v-if="'iou_label' in urec" class="unitlabel"
          >{{urec.iou_label}}</div></template>
<div class="info"
 ><span class="layername">{{urec.layer}}</span
 > <span class="unitnum">{{urec.unit}}</span>
</div>
<div class="thumbcrop"
><img
 :src="urec.dirname + '/s-image/' + urec.unit + '-top.jpg'"
 height="72"
></div>

<div class="ablationmark"></div>
<div class="insertionmark"></div>
</div> <!-- end unit -->

</div> <!-- end unitgrid -->

</div> <!-- end palette -->

</td>
</tr>
</table> <!-- end pane -->

<table class="pane" id="rightpane">
<tr>
<td rowspan="2" class="layout">

<div class="preview" v-if="dissect">
<p>Seeds to generate <input size="30" v-model="image_numbers"></p>
<p style="text-align: left">
To transfer activations from one pixel to another (1) click on a source pixel
on the left image and (2) click on a target pixel on a right image,
then (3) choose a set of units to insert in the palette.</p>
<div v-for="ex in examples" class="examplepair">
<div class="exampleid">#{{ ex.id }}</div>
<img :src="ex.baseline" style="max-width:50%;"
     :data-imgid="ex.id"
     data-side="left"
     v-on:click="clickexample"
><img :src="ex.modified" style="max-width:50%;"
     :data-imgid="ex.id"
     data-side="right"
     v-on:click="clickexample"
><div class="querymark" v-if="ex.mask" :style="{
   top: ((ex.mask.bitbounds[0] + ex.mask.bitbounds[2]) / 0.02
          / ex.mask.shape[0]) + '%',
   left: ((ex.mask.bitbounds[1] + ex.mask.bitbounds[3]) / 0.04
          / ex.mask.shape[1]) + '%'}"
></div><div class="editmark" v-if="ex.edit" :style="{
   top: ((ex.edit.bitbounds[0] + ex.edit.bitbounds[2]) / 0.02
          / ex.edit.shape[0]) + '%',
   left: ((ex.edit.bitbounds[1] + ex.edit.bitbounds[3]) / 0.04
          / ex.edit.shape[1]) + 50 + '%'}"
></div>
</div>

</div>

</td>
</tr>
</table>

</div> <!-- end app -->
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$(document).on('click', '[data-toggle=lightbox]', function(event) {
    $('#lightbox img').attr('src', $(this).attr('href'));
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).data('footer') ||
       $(this).closest('.unit').find('.info').text());
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('mousedown', '.unit', function(event) {
  if (!(event.buttons & 1)) return;
  var layer = $(event.currentTarget).find('.info .layername').text();
  var unit = $(event.currentTarget).closest('.unit').data('ordernum');
  theapp.dragging.active = true;
  theapp.dragging.first = unit;
  theapp.dragging.last = unit;
  event.preventDefault();
});
$(document).on('mouseup', function(event) {
  if (!theapp.dragging.active) { return; }
  theapp.dragging.active = false;
  var first_sel = $('.unit').filter(function(i, elt) {
                     return elt.dataset.ordernum == theapp.dragging.first;
                  }).map(function(i, elt) {
                        return {layer: $(elt).find('.layername').text(),
                                unit: $(elt).find('.unitnum').text()}; })[0];
  var selected = $('.unit').filter(function(i, elt) {
         return (theapp.dragging.first <= elt.dataset.ordernum &&
                 elt.dataset.ordernum <= theapp.dragging.last) ||
                (theapp.dragging.last <= elt.dataset.ordernum &&
                 elt.dataset.ordernum <= theapp.dragging.first); })
                .map(function(i, elt) {
                        return {layer: $(elt).find('.layername').text(),
                                unit: $(elt).find('.unitnum').text()}; });
  if (selected.length) {
    var selection = 'selected_' + theapp.selection_mode;
    var mode = 1 - theapp[selection][first_sel.layer][first_sel.unit];
    for (u of selected) {
      theapp[selection][u.layer].splice(u.unit, 1, mode);
    }
    theapp.selectionchange();
  }
});
$(document).on('mouseenter', '.unit', function(event) {
  if (!(event.buttons & 1)) { theapp.dragging.active = false; }
  if (!theapp.dragging.active) return;
  theapp.dragging.last = 
      $(event.currentTarget).closest('.unit').data('ordernum');
});
$(function() {
  window.Split(['#leftpane', '#rightpane'], {
              sizes: [50, 50],
              minSize: 280,
              snapOffset: 0,
  });
});
var theapp = new Vue({
  el: '#app',
  data: {
    palette_mode: 'ranking',
    sort_order: 'unit',
    palette_layer: null,
    selected_ablation: null,
    selected_insertion: null,
    selection_mode: 'ablation',
    dragging: { active: false, first: null, last: null },
    recipe: null,
    dissect: null,
    image_numbers: '10-19',
    examples: _.range(10, 20).map(function(x) {
        return {id: x, baseline: '', modified: '', mask: null, edit: null}; }),
    query_stat: 'mean_quantile',
    query_ranking: {},
  },
  created: function() {
    var self = this;
    $.getJSON('dissect.json?' + Math.random(), function(d) {
      self.selected_ablation = {};
      self.selected_insertion = {};
      for (var layer of d.layers) {
        // Preprocess ranking records to sort them.
        for (var rank of layer.rankings) {
          if (!('ranking' in rank)) {
            rank.ranking = rank.score.map((score, index) => [score, index])
                .sort(([score1], [score2]) => score1 - score2)
                .map(([, index]) => index);
          }
        }
        // Note layer in each unit record to simplify mixing.
        for (var urec of layer.units) {
          urec.layer = layer.layer;
          urec.dirname = layer.dirname;
        }
        Vue.set(self.selected_ablation, layer.layer,
                        _.fill(Array(layer.units.length), 0));
        Vue.set(self.selected_insertion, layer.layer,
                        _.fill(Array(layer.units.length), 0));
      }
      self.dissect = d;
      self.sort_order = d.default_ranking;
      self.palette_layer = Math.floor((d.layers.length - 1) / 2);
    });
    this.selectionchange();
  },
  computed: {
    currentquery: function() {
      var regions = [];
      for (var ex of this.examples) {
        if (ex.mask) {
          regions.push({id: ex.id, mask: ex.mask});
        }
      }
      return regions;
    },
    palette_units: function() {
      if (this.palette_mode == 'ranking') {
        // Order units according to an iou-matching sort order.
        var lrec = this.dissect.layers[this.palette_layer];
        var ranking = _.find(lrec.rankings, x=>x.name == this.sort_order);
        return ranking.ranking.map(x => lrec.units[x]);
      } else if (this.palette_mode == 'ablation' ||
                 this.palette_mode == 'insertion') {
        // Show units involved in the edit
        var result = [];
        var selectionname = 'selected_' + this.palette_mode;
        for (var lrec of this.dissect.layers) {
          var sel = this[selectionname][lrec.layer];
          for (var u in sel) {
            if (sel[u] > 0) {
              result.push(lrec.units[u]);
            }
          }
        }
        return result;
      } else if (this.palette_mode == 'query') {
        // Order units according to query ranking
        var lrec = this.dissect.layers[this.palette_layer];
        var ranking = this.query_ranking[lrec.layer];
        return ranking.ranking.map(x => lrec.units[x]);
      }
    }
  },
  watch: {
    query_stat: function() {
      this.querychange();
    },
    palette_layer: function(val) {
      // If sort_order is not available at this layer, reset it to default.
      var self = this;
      if (!_.find(self.dissect.layers[val].rankings,
            function(x) { return x.name == self.sort_order; })) {
        self.sort_order = self.dissect.default_ranking;
      }
    },
    image_numbers: function(val) {
      // Parse a series of image numbers
      var max_examples = 1000;
      var rs = val.replace(/[^-\d]+/g, ' ').replace(/\s*-\s*/g, '-').split(' ');
      var indexes = [];
      for (var r of rs) {
        if (r.match(/\d+-\d+/)) {
          for (var i = parseInt(r.split('-')[0]);
                          i <= parseInt(r.split('-')[1]); i++) {
            indexes.push(i);
          }
        } else if (r.match(/\d+/)) {
          indexes.push(parseInt(r));
        } else if (indexes.length == 0) {
          indexes.push(0);
        }
        if (indexes.length >= max_examples) { break; }
      }
      // Update examples to match.
      var modified = false;
      var examples = this.examples;
      for (i = 0; i < indexes.length; i++) {
        if (i >= examples.length) {
          examples.push({id: indexes[i], baseline: '', modified: '',
             mask: null });
          modified = true
        } else if (examples[i].id != indexes[i]) {
          examples[i].id = indexes[i];
          examples[i].baseline = '';
          examples[i].modified = '';
          examples[i].mask = null;
          examples[i].edit = null;
          modified = true
        }
      }
      if (examples.length > indexes.length) {
        examples.splice(indexes.length, examples.length - indexes.length);
        modified = true
      }
      if (modified) {
        this.generate_examples([], _.range(examples.length), true);
        this.selectionchange();
      }
    }
  },
  methods: {
    hashchange: function() {
      this.palette_layer = +window.location.hash.substr(1) || 0;
    },
    resetselection: function(mode) {
      var selection = 'selected_' + (mode);
      console.log(selection);
      for (var layer in this[selection]) {
        this[selection][layer] = _.fill(Array(this[selection][layer].length),0);
      }
      this.selectionchange();
    },
    invertselection: function(mode) {
      var layer = this.dissect.layers[this.palette_layer].layer;
      var selection = 'selected_' + (mode);
      var sel = this[selection][layer];
      for (var u in sel) {
        sel.splice(u, 1, 1 - sel[u]);
      }
      this.selectionchange();
    },
    selectionchange: function() {
      var edited_indices = [];
      var unedited_indices = [];
      var ca = this.currentablations();
      for (var i in this.examples) {
        var ci = this.currentinsertion(ca, i);
        if (ci) {
          edited_indices.push({intervention: ci, index: i});
        } else{
          unedited_indices.push(i);
        }
      }
      if (!window.skipupdate && unedited_indices.length) {
        this.generate_examples(ca, unedited_indices, false);
      }
      for (var r of edited_indices) {
        this.generate_examples(r.intervention, [r.index], false);
      }
    },
    currentablations: function() {
      var ablations = [];
      if (this.selected_ablation) {
        for (var layer in this.selected_ablation) {
          for (var unit in this.selected_ablation[layer]) {
            if (this.selected_ablation[layer][unit] > 0) {
              ablations.push({
                  layer: layer,
                  unit: parseInt(unit),
                  alpha: this.selected_ablation[layer][unit],
              });
            }
          }
        }
      }
      return ablations.length ? [{ablations: ablations}] : [];
    },
    currentinsertion: function(currentablations, exindex) {
      if (!this.examples[exindex].edit || !this.selected_insertion) {
        return null;
      }
      var insertions = [];
      for (var layer in this.selected_insertion) {
        for (var unit in this.selected_insertion[layer]) {
          if (this.selected_insertion[layer][unit] > 0) {
            insertions.push({
                layer: layer,
                unit: parseInt(unit),
                alpha: this.selected_insertion[layer][unit],
                value: this.query_ranking && layer in this.query_ranking &&
                       this.query_ranking[layer].activation[unit] || 0
            });
          }
        }
      }
      var result = _.clone(currentablations);
      result.push({ablations: insertions, mask: this.examples[exindex].edit});
      return result;
    },
    querychange: function() {
      this.query_regions(this.currentquery, this.currentablations());
    },
    query_regions: function(regions, intervention) {
      if (!_.keys(regions).length) {
        return;
      }
      var ids = regions.map(x => x.id);
      var masks = regions.map(x => x.mask);
      var self = this;
      $.post({
        url: '/api/features',
        data: JSON.stringify({
          project: this.currentproject(),
          ids: ids,
          masks: masks,
          // layers: [this.dissect.layers[this.palette_layer].layer],
          layers: this.dissect.layers.map(x => x.layer),
          // interventions: intervention,
        }),
        headers: {
          "Content-type": "application/json; charset=UTF-8"
        },
        success: function(resp) {
          var statname = self.query_stat;
          var actname = statname.replace('_quantile', '');
          var stats = resp.res;
          var result = {};
          for (var layer in stats) {
            result[layer] = {
              score: stats[layer][statname],
              ranking: stats[layer][statname]
                .map((score, index) => [score, index])
                .sort(([score1], [score2]) => score2 - score1)
                .map(([, index]) => index),
              activation: stats[layer][actname]};
          }
          self.query_ranking = result;
          self.palette_mode = 'query';
          // If there are any insertions, now the edit has changed
          if (_.sum(_.map(self.selected_insertion, x => _.sum(x))) > 0) {
            self.selectionchange();
          }
        },
        dataType: 'json'
      });
    },
    clickexample: function(ev) {
      var elt = ev.currentTarget;
      var imgid = elt.dataset.imgid * 1;
      var side = elt.dataset.side;
      var w = elt.naturalWidth;
      var h = elt.naturalHeight;
      var x = Math.round((ev.pageX - $(elt).offset().left) * (w / elt.width));
      var y = Math.round((ev.pageY - $(elt).offset().top) * (h / elt.height));
      // Clear all masks and leave one mask with one pixel chosen.
      var field = (side == 'right') ? 'edit' : 'mask'
      for (var ex of this.examples) {
        if (ex.id != imgid) {
          Vue.set(ex, field, null);
        } else {
          Vue.set(ex, field, {
            shape: [h, w],
            bitbounds: [y, x, y+1, x+1],
            bitstring: '1'
          });
        }
      }
      if (field == 'edit') {
        this.selection_mode = 'insertion';
        this.selectionchange();
      } else {
        this.querychange();
      }
    },
    generate_examples: function(intervention, example_indexes, baseline_only) {
      var self = this;
      var ids = example_indexes.map(x => self.examples[x].id);
      $.post({
        url: '/api/generate',
        data: JSON.stringify({
          project: this.currentproject(),
          wantz: 0,
          ids: ids,
          interventions: intervention
        }),
        headers: {
          "Content-type": "application/json; charset=UTF-8"
        },
        success: function(resp) {
          for (var j in resp.res) {
            var i = example_indexes[j];
            if (self.examples[i].id == ids[j]) {
              if (!baseline_only) {
                self.examples[i].modified = resp.res[j].d;
              }
              if (!intervention || !intervention.length) {
                self.examples[i].baseline = resp.res[j].d;
              }
            }
          }
        },
        dataType: 'json'
      });
    },
    currentproject: function() {
      return location.pathname.match(/\/data\/([^\/]*)/)[1];
    }
  },
  filters: {
    fixed: function(value, digits, truncate) {
       if (typeof value != 'number') return value;
       var fixed = value.toFixed(digits);
       return truncate ? +fixed : fixed;
    }
  }
});
</script>
</body>
</html>
