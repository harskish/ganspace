


var sliders = document.getElementsByClassName("slider");
var slider_name;
var slider_value;
var layer_start_value;
var layer_end_value;
var component_sliders = [];

// var xyro_refresh_timer = setInterval(xyro_refresh_function, 250);

function xyro_refresh_function(){

    //refreshes an image with a #dummy_image class regardless of caching
    //get the src attribute
    source = jQuery("#dummy_image").attr("src");
    //remove previously added timestamps
    source = source.split("?", 1);//turns "image.jpg?timestamp=1234" into "image.jpg" avoiding infinitely adding new timestamps
    //prep new src attribute by adding a timestamp
    new_source = source + "?timestamp="  + new Date().getTime();
    //alert(new_source); //you may want to alert that during developement to see if you're getting what you wanted
    //set the new src attribute
    jQuery("#dummy_image").attr("src", new_source);
}

for(var i=0; i<(sliders.length); i++) {

    //initialize sliders
    slider_name = sliders[i].getAttribute('name');
    slider_value = sliders[i].value;
    document.getElementById(sliders[i].getAttribute('id')+'_value').innerHTML = slider_value;
        if (slider_name == 'layer_start'){
            layer_start_value = slider_value;
        } else if(slider_name == 'layer_end'){
            layer_end_value = slider_value;
        } else if(slider_name == 'component_slider'){
            component_sliders.push(slider_value); //initialize component slider array
        }


    //Update sliders and send it to server
    sliders[i].addEventListener('input', function() {
        slider_name = this.getAttribute('name');
        slider_value = this.value;
        
        
        //console.log(slider_name);

        document.getElementById(this.getAttribute('id')+'_value').innerHTML = slider_value;

        if (slider_name == 'layer_start'){
            layer_start_value = slider_value;
        } else if(slider_name == 'layer_end'){
            layer_end_value = slider_value;
        } else if(slider_name == 'component_slider'){
            index = this.getAttribute('id').split("_")[1];
            component_sliders[index] = slider_value; //overwrites component slider values
        }
        var seed_value = $(".seed_value").val(); 

        $.ajax({
            method: 'POST',
            dataType: 'json',
            data: {
                'type': 'slider_value_update',
                'model_name': model_name,
                'layer_start': layer_start_value,
                'layer_end': layer_end_value,
                'component_sliders[]': component_sliders,
                'seed': seed_value,
                csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
            success: function(data){
                console.log(layer_start_value)
                xyro_refresh_function()
            }
        });
        
    });
}

$(".resample_button").on("click", function(){
    $.ajax({
        method: 'POST',
        // dataType: 'json',
        data: {
            'type': 'resample_latent',
            'layer_start': layer_start_value,
            'layer_end': layer_end_value,
            'component_sliders[]': component_sliders,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
        success: function(data){
            xyro_refresh_function()
            $(".seed_value").val(data['seed']);
        }
    });
});


$(".update_button").on("click", function(){
    var seed_value = $(".seed_value").val(); 
    $.ajax({
        method: 'POST',
        // dataType: 'json',
        data: { 
            'type': 'update_seed',
            'layer_start': layer_start_value,
            'layer_end': layer_end_value,
            'component_sliders[]': component_sliders,
            'seed': seed_value,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
        success: function(data){
            console.log(layer_start_value)
            xyro_refresh_function()
        }
    });
});
