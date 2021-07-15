import os

try:
    # if requirements.txt is absent or it is blank then we get error while uninstalling it
    if os.path.exists('requirements.txt'):
        os.system('cp requirements.txt gpu_requirements.txt')
    os.system('pip3 freeze > requirements.txt && pip3 uninstall -r requirements.txt -y 2> /dev/null') #not displaying error message if requirements.txt is empty
    os.remove('requirements.txt')
except:
    #if we don't get uninstallation error. It means we have uninstallation step was done correctly
    pass

# now we need to install the packages from yolov5 repository
os.system('cp yolov5/requirements.txt requirements.txt')
os.system('pip3 install -qr requirements.txt')
