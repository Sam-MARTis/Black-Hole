#pragma once

#include <iostream>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float2 camera = {0.1f, 1.5f};
const float pi = 3.1415926535f;
const float twopi = 2*pi;



void print(const auto& message){
    std::cout<<message<<"\n";
}

void printInstructions(){
    print("Use WASD for moving");
}



const float epsilon = 0.001f;
void moveRight(){
    camera.x =  camera.x - 0.03;
    camera.x < epsilon ? (camera.x = twopi-epsilon) : 0;
}

void moveLeft(){
    camera.x += 0.03;
    camera.x > twopi ? (camera.x= epsilon): 0;
}

void moveUp(){
    camera.y+= 0.03;
    camera.y > pi ? (camera.y=pi - epsilon): 0;
}
void moveDown(){
    camera.y-= 0.03;
    camera.y < epsilon ? (camera.y= epsilon): 0;
}


void keyboard(unsigned char key, int x, int y) {
  if (key == 'a') moveLeft();
  if (key == 'd') moveRight();
  if(key=='w') moveUp();
  if(key=='s') moveDown();
//   print("Camera position: (" + std::to_string(camera.x) + ", " + std::to_string(camera.y) + ")");

  if (key == 27) exit(0);

  glutPostRedisplay();
}