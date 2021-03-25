$(document).ready(function(){

    $(".resample_button").on("click", function(){
        $.ajax({
            method: 'POST',
            // dataType: 'json',
            data: {csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()},
            success: function(data){
                console.log(data['seed'])
            }
        });
    });

});   