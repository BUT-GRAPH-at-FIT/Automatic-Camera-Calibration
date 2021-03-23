from calibration import Calibration
#for example load only
import json

if __name__=='__main__':

    with open('example_input.json', 'r') as f:
        objects = json.load(f)

    objects = objects[:10]

    # c = Calibration().calibrate(objects, method='Plane')
    # c = Calibration().calibrate(objects, method='Landmarks')

    print(c)
