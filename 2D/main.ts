
const canvas = document.getElementById("projectCanvas") as HTMLCanvasElement;
canvas.width = window.innerWidth*devicePixelRatio;
canvas.height = window.innerHeight*devicePixelRatio;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
if(!ctx){
    throw Error("Context unable to be found");
}




const STEP:number = 0.01;




class vector2D {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
    normalize(inplace: boolean = true) {
        const length = Math.sqrt(this.x * this.x + this.y * this.y);
        if(inplace) {
            this.x /= length;
            this.y /= length;
            return this;
        }
        else return new vector2D(this.x / length, this.y / length);
    }
    add(other: vector2D, inplace: boolean = true) {
        if (inplace) {
            this.x += other.x;
            this.y += other.y;
            return this;
        }

        else return new vector2D(this.x + other.x, this.y + other.y);
    }
    subtract(other: vector2D, inplace: boolean = true) {
        if (inplace) {
            this.x -= other.x;
            this.y -= other.y;
            return this;
        }
        else return new vector2D(this.x - other.x, this.y - other.y);
    }
    multiply(scalar: number, inplace: boolean = true) {
        if (inplace) {
            this.x *= scalar;
            this.y *= scalar;
            return this;
        }
        else return new vector2D(this.x * scalar, this.y * scalar);
    }
}
class Photon {
    pos: vector2D;
    direction: vector2D;


    constructor(x: number, y: number, vx: number, vy: number) {
        this.pos = new vector2D(x, y);
        this.direction = new vector2D(vx, vy).normalize();

    }

    step(){
        this.pos.add(this.direction.multiply(STEP, false));
    }

    draw(){
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.closePath();
    }
}




