from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.files import File
from backend.predictor.models import TrainedModel
import os
import shutil

class Command(BaseCommand):
    help = 'Migrate model files from old location to media directory'

    def handle(self, *args, **options):
        # Source and destination directories
        old_dir = os.path.join(settings.BASE_DIR, 'trained_models')
        new_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')

        # Create new directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)

        # Get all trained models
        models = TrainedModel.objects.all()
        
        for model in models:
            try:
                # Get model name from the file path
                model_name = os.path.basename(model.model_file.name)
                if not model_name.endswith('.h5'):
                    model_name = model_name + '.h5'
                
                # Get the old file paths
                model_dir = os.path.splitext(model_name)[0]
                old_model_dir = os.path.join(old_dir, model_dir)
                old_path = os.path.join(old_model_dir, model_name)
                
                # Get the new file path
                new_path = os.path.join(new_dir, model_name)
                
                # Copy the files if they exist
                if os.path.exists(old_path):
                    # Copy model file
                    shutil.copy2(old_path, new_path)
                    
                    # Copy scaler files if they exist
                    scaler_x = os.path.join(old_model_dir, 'scaler_X.pkl')
                    scaler_y = os.path.join(old_model_dir, 'scaler_y.pkl')
                    
                    if os.path.exists(scaler_x):
                        shutil.copy2(scaler_x, os.path.join(new_dir, 'scaler_X.pkl'))
                    if os.path.exists(scaler_y):
                        shutil.copy2(scaler_y, os.path.join(new_dir, 'scaler_y.pkl'))
                    
                    # Update the model record with the new path
                    with open(new_path, 'rb') as f:
                        model.model_file.save(model_name, File(f), save=True)
                    
                    self.stdout.write(self.style.SUCCESS(f'Successfully migrated {model_name}'))
                else:
                    self.stdout.write(self.style.WARNING(f'File not found: {old_path}'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error migrating {model.name}: {str(e)}'))

        self.stdout.write(self.style.SUCCESS('Model migration completed')) 