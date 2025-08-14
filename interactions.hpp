#pragma once

#include <iostream>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float3 camera = {3.0f, 0.0f, 1.5f};
const float pi = 3.1415926535f;
const float twopi = 2*pi;



void print(const auto& message){
    std::cout<<message<<"\n";
}

void printInstructions(){
    print("Use WASD for moving, mn for zoom");
}

#define SIGN(x) ((x) < 0 ? -1 : 1)

const float epsilon = 0.0001f;
void moveForward(){
    camera.x += 0.05;
}
void moveBackward(){
    camera.x -= 0.05;
}
void moveRight(){
    camera.y =  camera.y - 0.03;
    camera.y < epsilon ? (camera.y = twopi-epsilon) : 0;
}

void moveLeft(){
    camera.y += 0.03;
    camera.y > twopi ? (camera.y= epsilon): 0;
}


void moveUp(){
    camera.z+= 0.03;
    // camera.z > pi-epsilon ? (camera.z=epsilon): 0;
    // camera.z > pi-epsilon ? (camera.z= -(pi-epsilon)): 0;
    // camera.z > (pi-epsilon) ? ({camera.y += pi; camera.z = epsilon; }): 0;
    camera.z > (pi-epsilon) ? (camera.z = (epsilon - pi)): 0;
}
void moveDown(){
    camera.z-= 0.03;

    camera.z < epsilon-pi ? (camera.z = pi-epsilon): 0;
    // camera.z > (pi-epsilon) ? (camera.y += pi): 0;
    // camera.z < epsilon ? (camera.z= pi-epsilon): 0;
    // camera.z < -(pi-epsilon) ? (camera.z= pi-epsilon): 0;
    // camera.z > pi-epsilon ? (camera.z=epsilon): 0
    

    // camera.z < epsilon ? (camera.z= pi-epsilon): 0;
    // camera.z < -(pi-epsilon) ? (camera.z= pi-epsilon): 0;
}


void keyboard(unsigned char key, int x, int y) {
  if (key == 'a') moveLeft();
  if (key == 'd') moveRight();
  if(key=='w') moveUp();
  if(key=='s') moveDown();
    if (key == 'm') moveForward();
    if (key == 'n') moveBackward();
//   print("Camera position: (" + std::to_string(camera.y) + ", " + std::to_string(camera.z) + ")");

  if (key == 27) exit(0);

  glutPostRedisplay();
}