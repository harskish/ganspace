<!DOCTYPE html>
<html>
<!--
  +lightbox.html, a page for automatically showing all images in a
  directory on an Apache server. Just copy it into the directory.
  Works by scraping the default directory HTML at "./" - David Bau.
-->
<head>
<script src="https://cdn.jsdelivr.net/npm/vue@2.5.16/dist/vue.js"
  integrity="sha256-CMMTrj5gGwOAXBeFi7kNokqowkzbeL8ydAJy39ewjkQ="
  crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/lodash@4.17.10/lodash.js"
  integrity="sha256-qwbDmNVLiCqkqRBpF46q5bjYH11j5cd+K+Y6D3/ja28="
  crossorigin="anonymous"></script>
<script
  src="https://code.jquery.com/jquery-3.3.1.js"
  integrity="sha256-2Kok7MbOyxpgUVvAk/HJ2jigOSYS2auK4Pfzbm7uH60="
  crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lity/2.3.1/lity.js"
  integrity="sha256-28JiZvE/RethQIYCwkMdtSMHgI//KoTLeB2tSm10trs="
  crossorigin="anonymous"></script>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/lity/2.3.1/lity.css"
  integrity="sha256-76wKiAXVBs5Kyj7j0T43nlBCbvR6pqdeeZmXI4ATnY0="
  crossorigin="anonymous" />
<style>
.thumb { display: inline-block; margin: 1px; text-align: center; }
.thumb img { max-width: 150px; }
</style>
</head>
<body>
<div id="app" v-if="images">
<h3>Images in <a :href="directory">{{ directory }}</a></h3>
<div v-for="r in images" class="thumb">
<div>{{ r }}</div>
<a :href="r" data-lity><img :src="r"></a>
</div>
</div><!--app-->
</body>
<script>
var theapp = new Vue({
  el: '#app',
  data: {
    directory: window.location.pathname.replace(/[^\/]*$/, ''),
    images: null
  },
  created: function() {
    var self = this;
    $.get('./?' + Math.random(), function(d) {
      var imgurls = $.map($(d).find('a'),
                      x => x.href).filter(
                      x => x.match(/\.(jpg|jpeg|png|gif)$/i)).map(
                      x => x.replace(/.*\//, ''));
      self.images = imgurls;
    }, 'html');
  }
})
</script>
</html>
