// GET THE IMAGE.

//console.log(DJANGO_STATIC_URL);
// var img = new Image();
// img.src = img_src
// // WAIT TILL IMAGE IS LOADED.
// function draw_image(){
    
    
//     img.onload = function () {
//         fill_canvas(img);       // FILL THE CANVAS WITH THE IMAGE.
//     }
// }


// function fill_canvas(img) {
//     //alert("image has loaded")
//     // CREATE CANVAS CONTEXT.
//     var canvas = document.getElementById('image_canvas');
//     var ctx = canvas.getContext('2d');
//     canvas.width = img.width;
//     canvas.height = img.height;
//     ctx.drawImage(img, 0, 0); // DRAW THE IMAGE TO THE CANVAS.
// }

// draw_image();
var sliders = document.getElementsByClassName("slider");
for(var i=0; i<(sliders.length); i++) {
    //update image everytime slider is modified
    sliders[i].addEventListener('input', function() {
        $.ajax({
            method: 'POST',
            dataType: 'json',
            data: {
            'type': 'get_new_image',
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
            success: function(data){
                console.log(data['seed'])
            }
        });
        
    });
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