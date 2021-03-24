// GET THE IMAGE.
var img = new Image();
img.src = img_src
//console.log(DJANGO_STATIC_URL);

// WAIT TILL IMAGE IS LOADED.
img.onload = function () {
    fill_canvas(img);       // FILL THE CANVAS WITH THE IMAGE.
}

function fill_canvas(img) {
    //alert("image has loaded")
    // CREATE CANVAS CONTEXT.
    var canvas = document.getElementById('image_canvas');
    var ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0); // DRAW THE IMAGE TO THE CANVAS.
}

// function testImage(URL) {
//     var tester=new Image();
//     tester.src=URL;
//     tester.onload=imageFound;
//     tester.onerror=imageNotFound;
    
// }

// function imageFound() {
//     alert('That image is found and loaded');
// }

// function imageNotFound() {
//     alert('That image was not found.');
// }

//testImage(img_src);