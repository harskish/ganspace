from django.db import models

# Create your models here.
class ganspace_model(models.Model):
    className = models.CharField(max_length = 50)
    number_of_components = models.IntegerField(default=70)
    hide_model = models.BooleanField(default=False)

    def __str__(self):
        return self.className
    

class model_properties(models.Model):
    className = models.ForeignKey(ganspace_model, null=True ,on_delete=models.SET_NULL)
    #component_values = models.FilePathField()#TODO edit the file path
    seed = models.IntegerField()


