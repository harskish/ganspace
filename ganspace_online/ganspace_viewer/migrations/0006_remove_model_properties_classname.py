# Generated by Django 3.1.7 on 2021-03-22 19:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ganspace_viewer', '0005_auto_20210322_1553'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='model_properties',
            name='className',
        ),
    ]
