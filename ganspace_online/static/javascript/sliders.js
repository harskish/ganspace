


var sliders = document.getElementsByClassName("slider");
var slider_name;
var slider_value;
var layer_start_value;
var layer_end_value;
var component_sliders = [];

//component_sliders.push("1");

// for (var comp_number in component_slider_list){
//     document.getElementById('component_name_'+comp_number).innerHTML =  component_slider_number;
// }


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
        
        $.ajax({
            method: 'POST',
            dataType: 'json',
            data: {
                'type': 'slider_value_update',
                'layer_start': layer_start_value,
                'layer_end': layer_end_value,
                'component_sliders[]': component_sliders,
                csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
            success: function(data){
                console.log(data['seed'])
                $('#dummy_image').attr('src', img_src+ '?'+slider_value);
            }
        });
        
    });
}





