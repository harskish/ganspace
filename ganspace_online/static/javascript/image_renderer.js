window.onload = function () {
    // GET THE IMAGE.
    var img = new Image();
    var url = '../media/test.jpg';
    //img.attr('src',DJANGO_STATIC_URLL+'/media/test.jpg');
    img.src = DJANGO_STATIC_URL+'media/test.jpg'
    //console.log(DJANGO_STATIC_URL);

    // WAIT TILL IMAGE IS LOADED.
    img.onload = function () {
        fill_canvas(img);       // FILL THE CANVAS WITH THE IMAGE.
    }

    function fill_canvas(img) {
        // CREATE CANVAS CONTEXT.
        var canvas = document.getElementById('image_canvas');
        var ctx = canvas.getContext('2d');
        // canvas.width = img.width;
        // canvas.height = img.height;

        ctx.drawImage(img, 0, 0);       // DRAW THE IMAGE TO THE CANVAS.
    }
}