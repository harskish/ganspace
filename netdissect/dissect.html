<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.3.1.js" integrity="sha256-2Kok7MbOyxpgUVvAk/HJ2jigOSYS2auK4Pfzbm7uH60=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vue/2.5.16/vue.js" integrity="sha256-CMMTrj5gGwOAXBeFi7kNokqowkzbeL8ydAJy39ewjkQ=" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/lodash@4.17.10/lodash.js" integrity="sha256-qwbDmNVLiCqkqRBpF46q5bjYH11j5cd+K+Y6D3/ja28=" crossorigin="anonymous"></script>
<style>
[v-cloak] {
  display: none;
}
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 288px;
  height: 72px;
}
.thumbcrop img, .img-scroller img {
  image-rendering: pixelated;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
  margin-left: 5px;
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
  position: relative;
}
.img-mask, .img-seg {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 0;
  visibility: hidden;
}
input.hidden-toggle {
  display: none;
}
#show-seg:checked ~ .img-wrapper .img-seg,
#show-mask:checked ~ .img-wrapper .img-mask {
  visibility: visible;
}
.img-controls {
  text-align: right;
}
.img-controls label {
  display: inline-block;
  background: silver;
  padding: 10px;
  margin-top: 0;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.seginfo {
  display: inline-block;
  padding: 10px;
  float: left;
}
.img-mask {
  pointer-events: none;
}
.colorsample {
  display: inline-block;
  height: 42px;
  width: 42px;
  float: left;
}
#show-seg:checked ~ .img-controls .toggle-seg,
#show-mask:checked ~ .img-controls .toggle-mask {
  background: navy;
  color: white;
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
.dropmenu {
  display: inline-block;
  vertical-align: top;
  position: relative;
}
.dropmenulist {
  pointer-events: auto;
  visibility: hidden;
  transition: visiblity 1s;
  position: absolute;
  z-index: 1;
  background: white;
  right: 0;
  text-align: right;
  white-space: nowrap;
}
.dropmenu:focus {
  pointer-events: none;
}
.dropmenu:focus .dropmenulist {
  visibility: visible;
}
</style>
</head>
<body class="unitviz">
<div id="app" v-if="dissect" v-cloak>

<nav class="navbar navbar-expand navbar-dark bg-inverse">
<span class="navbar-brand">{{ dissect.netname || 'Dissection' }}</span>
<ul class="navbar-nav mr-auto">
<li :class="{'nav-item': true, active: lindex == selected_layer}"
    v-for="(lrec, lindex) in dissect.layers">
 <a class="nav-link" :href="'#' + lindex"
    >{{lrec.layer}}</a>
</li>
</ul>
<ul class="navbar-nav ml-auto" v-if="dissect.meta">
  <li class="navbar-text ml-2" v-for="(v, k) in dissect.meta">
    {{k}}={{v | fixed(3, true)}}
  </li>
</ul>
</nav>

<div v-for="lrec in [dissect.layers[selected_layer]]">
<div v-if="'bargraph' in lrec" class="histogram">
<a data-toggle="lightbox" :href="lrec.dirname + '/bargraph.svg?'+Math.random()"
   :data-title="'Summary of ' + (dissect.netname || 'labels')
          + ' at ' + lrec.layer">
<img class="img-fluid"
   :src="lrec.dirname + '/' + lrec.bargraph + '?'+Math.random()">
</a>
</div>

<div class="gridheader">
<div class="layerinfo">
<span v-if="'interpretable' in lrec"
>{{lrec.interpretable}}/</span
>{{lrec.units.length}} units
<span v-if="'labels' in lrec">
covering {{lrec.labels.length}} concepts
with IoU &ge; {{dissect.iou_threshold}}
</span>
</div>

<div class="sortheader">
sort by
<span v-for="rank in lrec['rankings']" v-if="!rank.metric">
<span :class="{sortby: true, currentsort: sort_order == rank.name}"
      :data-ranking="rank.name"
      v-on:click="sort_order = $event.currentTarget.dataset.ranking"
      >{{rank.name}}</span>
<span> </span>
</span>
<span v-for="metric in _.filter(_.uniq(lrec.rankings.map(x => x.metric)))">
  <div class="dropmenu sortby" tabindex="0">
    <div class="dropmenutop">
      *-{{ metric }}
    </div>
    <div class="dropmenulist">
        <div v-for="rank in lrec['rankings']" v-if="rank.metric == metric">
        <span :class="{sortby: true, currentsort: sort_order == rank.name}"
              :data-ranking="rank.name"
              v-on:click="sort_order = $event.currentTarget.dataset.ranking"
              >{{rank.name}}</span>
        </div>
    </div>
  </div>
  <span> </span>
</span>

</div>

</div>
<div class="unitgrid"
      v-for="lk in [_.find(lrec.rankings, x=>x.name == sort_order)
                     .metric || 'iou']"
><div :class="{unit: true, lowscore: lk == 'iou' && !urec.interp}"
      v-for="urec in _.find(lrec.rankings, x=>x.name == sort_order)
                      .ranking.map(x=>lrec.units[x])">
<div v-if="lk+'_label' in urec" class="unitlabel">{{urec[lk+'_label']}}</div>
<div class="info"
 ><span class="layername">{{lrec.layer}}</span
 > <span class="unitnum">unit {{urec.unit}}</span
 > <span v-if="lk+'_cat' in urec" class="category">({{urec[lk+'_cat']}})</span
 > <span v-if="lk+'_iou' in urec" class="iou"
         >iou {{urec[lk + '_iou'] | fixed(2)}}</span
 > <span v-if="lk in urec" class="iou"
         >{{lk}} {{urec[lk] | fixed(2)}}</span></div>
<div class="thumbcrop" v-for="imprefix in [lrec['image_prefix_' + lk] || '']"
><a data-toggle="lightbox"
 :href="lrec.dirname + '/' + imprefix + 'image/' + urec.unit + '-top.jpg'"
><img
 :src="lrec.dirname + '/' + imprefix + 'image/' + urec.unit + '-top.jpg'"
 height="72"></a></div>
</div></div> <!-- end unit -->

</div> <!-- end unit grid -->

</div> <!-- end container -->

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
        <input id="show-seg" class="hidden-toggle" type="checkbox">
        <input id="show-mask" class="hidden-toggle" type="checkbox" checked>
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid img-orig">
          <img class="fullsize img-fluid img-seg">
          <img class="fullsize img-fluid img-mask">
        </div>
        <div class="img-controls">
          <canvas class="colorsample" height=1 width=1></canvas>
          <div class="seginfo">
          </div>
          <label for="show-seg" class="toggle-seg">segmentation</label>
          <label for="show-mask" class="toggle-mask">mask</label>
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
    if ($(this).attr('href').match(/-top/)) {
        $('#lightbox img.img-orig').attr('src',
            $(this).attr('href').replace(/-top.jpg/, '-orig.jpg'));
        $('#lightbox img.img-seg').attr('src',
            $(this).attr('href').replace(/-top.jpg/, '-seg.png'));
        $('#lightbox img.img-mask').attr('src',
            $(this).attr('href').replace(/-top.jpg/, '-mask.png'));
        $('#lightbox .img-seg, #lightbox .img-mask, .img-controls').show();
    } else {
        $('#lightbox img.img-orig').attr('src', $(this).attr('href'));
        $('#lightbox .img-seg, #lightbox .img-mask, .img-controls').hide();
    }
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).data('footer') ||
       $(this).closest('.unit').find('.info').text());
    $('#lightbox .segcolors').text('');
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('click', '#lightbox img.img-seg', function(event) {
  var elt_pos = $(this).offset();
  var img_x = event.pageX - elt_pos.left;
  var img_y = event.pageY - elt_pos.top;
  var canvas = $('#lightbox .colorsample').get(0);
  canvas.getContext('2d').drawImage(this, img_x, img_y, 1, 1, 0, 0, 1, 1);
  var pixelData = canvas.getContext('2d').getImageData(0, 0, 1, 1).data;
  var colorkey = pixelData[0] + ',' + pixelData[1] + ',' + pixelData[2];
  var meaning = theapp.dissect.segcolors[colorkey];
  $('#lightbox .seginfo').text(meaning);
});

var theapp = new Vue({
  el: '#app',
  data: {
    sort_order: 'unit',
    sort_fields: {
      label: [[], []],
      score: [['iou'], ['desc']],
      unit:  [['unit'], ['asc']],
    },
    selected_layer: null,
    dissect: null
  },
  created: function() {
    var self = this;
    $.getJSON('dissect.json?' + Math.random(), function(d) {
      self.dissect = d;
      for (var layer of d.layers) {
        // Preprocess ranking records to sort them.
        for (var rank of layer.rankings) {
          if (!('ranking' in rank)) {
            rank.ranking = rank.score.map((score, index) => [score, index])
                .sort(([score1], [score2]) => score1 - score2)
                .map(([, index]) => index);
          }
        }
      }
      self.sort_order = d.default_ranking;
      self.hashchange();
    });
    $(window).on('hashchange', function() { self.hashchange(); });
  },
  methods: {
    hashchange: function() {
      this.selected_layer = +window.location.hash.substr(1) || 0;
    },
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
