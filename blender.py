import bpy
from mathutils import Vector

import numpy as np 

import time

from multiprocessing import Process, Pool

def render(bpy, cam, z, y, theta, omega):
        bpy.context.scene.camera = cam
        
        x = y*np.sin(omega)
        k = y*np.cos(omega)
            
        cam.location = Vector([x, -k, z])

        cam.rotation_euler[0] = theta
        if theta <= 0:
            cam.rotation_euler[1] = np.pi
            
    #    theta = np.arctan(cam.location.y/cam.location.z)   
        
        bpy.context.scene.render.filepath = "/datasets/airplanes/testA/angle-{}-{}.jpg".format(omega,theta)
        bpy.ops.render.render(write_still = True)


if __name__ == '__main__':
    s = time.time()
    
    origin = Vector([0.0,0.0,0.0])

    obj = bpy.context.scene.objects[0]
#    cam = bpy.context.scene.objects[2]

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')

    radius = 60
    N = 12
    start_angles = np.array([0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2]) + np.pi/6
    cam = []

    for j, omega in enumerate(start_angles):
    
        cam_objects = [int(x.name.split(" ")[1])+1 for x in bpy.data.objects if x.name.startswith("cam")]
        
        if len(cam_objects) != 0:
            max_cam = np.max(cam_objects)
            if N > max_cam:
                num_cams = max_cam - N
            else:
                num_cams = 0
        else:
            num_cams = N

        for i in range(N):
            #if num_cams != 0 and (N-i-1) < num_cams:
            tmp_cam = bpy.data.cameras.new("cam {}".format(i+j*N))
            cam.append(bpy.data.objects.new("cam_obj {}".format(i+j*N), tmp_cam))
            bpy.context.scene.collection.objects.link(cam[i+j*N])
            #else:
            #    cam.append(bpy.context.scene.objects["cam_obj {}".format(i)])
            
            x = radius*np.sin(omega)
            y = radius*np.cos(omega)

            cam[i+j*N].location = Vector([x, -y, 0.0])
            cam[i+j*N].rotation_euler[0] = np.pi/2
            cam[i+j*N].rotation_euler[1] = 0
            cam[i+j*N].rotation_euler[2] = omega

    obj.location = origin

    angle = np.linspace(0,2*np.pi,N)
    z = [radius*np.sin(a) for a in angle]
    y = [radius*np.cos(a) for a in angle]
    thetas = [np.pi/2-a for a in angle]

    for i, val in enumerate(thetas):
        if val <= 0:
            thetas[i] = -np.pi - thetas[i]
        if val <= -np.pi:
            thetas[i] = np.pi - thetas[i]

    if N <= 1:
        for i in range(N):
            render(bpy, cam, z, y, thetas, omega, i)
    else:
        p = []
        j = 0    
        for i in range(len(cam)):
            p.append(Process(target=render, args=(bpy, cam[i], z[j], y[j], thetas[j], start_angles[int(i/N)])))
            j += 1
            if j == N:
                j = 0
        j = 0
        for i in range(len(cam)):
            p[i].start()
            j += 1
            if j == N:
                j = 0
        j = 0
        for i in range(len(cam)):
            p[i].join()
            j += 1
            if j == N:
                j = 0

    print (time.time() -s)