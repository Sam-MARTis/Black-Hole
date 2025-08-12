
const canvas = document.getElementById("projectCanvas") as HTMLCanvasElement;
canvas.width = window.innerWidth*devicePixelRatio;
canvas.height = window.innerHeight*devicePixelRatio;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
if(!ctx){
    throw Error("Context unable to be found");
}




const STEP:number = 1;
const PHOTON_RADIUS:number = 1;




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
        ctx.arc(this.pos.x, this.pos.y, PHOTON_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.closePath();
    }
}

class BlackHole{
    pos: vector2D;
    radius: number;

    constructor(x: number, y: number, radius: number) {
        this.pos = new vector2D(x, y);
        this.radius = radius;
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.closePath();
    }
}


const numberOfPhotons = 50;
let photons: Photon[] = [];
const init = () => {
    for(let i = 0; i<numberOfPhotons; i++){
        // const x = Math.random() * canvas.width;
        const x = 0;
        // const y = Math.random() * canvas.height;
        const y = (i+0.5)* canvas.height / numberOfPhotons
        const vx = 100*(Math.random() ) * 2;
        const vy = 0;
        // const vy = 100*(Math.random() - 0.5) * 2;
        photons.push(new Photon(x, y, vx, vy));
    }
    mainLoop();
}


const mainLoop = () => {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    photons.forEach(photon => {
        photon.step();
        photon.draw();
    });
    requestAnimationFrame(mainLoop);
}

window.addEventListener("load", init);



