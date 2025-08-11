#pragma once

#include <iostream>


float2 camera = {0.0f, 0.0f};
const float pi = 3.1415926535f;
const float twopi = 2*pi;



void print(char* message){
    std::cout<<message<<"\n";
}

void printInstructions(){
    print("Use WASD for moving");
}




void moveRight(){
    camera.x =  camera.x - 0.01;
    camera.x <0 ? (camera.x=twopi): 0;
}

void moveLeft(){
    camera.x += 0.01;
    camera.x > twopi ? (camera.x=0): 0;
}

void moveUp(){
    camera.y+= 0.01;
    camera.y > pi ? (camera.y=pi): 0;
}
void moveDown(){
    camera.y-= 0.01;
    camera.y < -pi ? (camera.y=-pi): 0;
}


void keyboard(unsigned char key, int x, int y) {
  if (key == 'a') moveLeft();
  if (key == 'd') moveRight();
  if(key=='w') moveUp();
  if(key=='s') moveDown();

  if (key == 27) exit(0);
  glutPostRedisplay();
}