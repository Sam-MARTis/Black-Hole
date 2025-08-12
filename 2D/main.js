"use strict";
const canvas = document.getElementById("projectCanvas");
canvas.width = window.innerWidth * devicePixelRatio;
canvas.height = window.innerHeight * devicePixelRatio;
const ctx = canvas.getContext("2d");
if (!ctx) {
    throw Error("Context unable to be found");
}
const STEP = 1;
const PHOTON_RADIUS = 1;
const numberOfPhotons = 5;
class vector2D {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    normalize(inplace = true) {
        const length = Math.sqrt(this.x * this.x + this.y * this.y);
        if (inplace) {
            this.x /= length;
            this.y /= length;
            return this;
        }
        else
            return new vector2D(this.x / length, this.y / length);
    }
    add(other, inplace = true) {
        if (inplace) {
            this.x += other.x;
            this.y += other.y;
            return this;
        }
        else
            return new vector2D(this.x + other.x, this.y + other.y);
    }
    subtract(other, inplace = true) {
        if (inplace) {
            this.x -= other.x;
            this.y -= other.y;
            return this;
        }
        else
            return new vector2D(this.x - other.x, this.y - other.y);
    }
    multiply(scalar, inplace = true) {
        if (inplace) {
            this.x *= scalar;
            this.y *= scalar;
            return this;
        }
        else
            return new vector2D(this.x * scalar, this.y * scalar);
    }
}
class Photon {
    constructor(x, y, vx, vy, blackHole) {
        this.alive = true;
        this.pos = new vector2D(x, y);
        this.direction = new vector2D(vx, vy).normalize();
        this.blackHole = blackHole;
        this.phi = Math.atan2(y - blackHole.pos.y, (x - blackHole.pos.x));
        this.r = Math.sqrt((x - blackHole.pos.x) ** 2 + (y - blackHole.pos.y) ** 2);
        this.dr = this.direction.x * Math.cos(this.phi) + this.direction.y * Math.sin(this.phi);
        this.dphi = -this.direction.x * Math.sin(this.phi) + this.direction.y * Math.cos(this.phi);
    }
    step() {
        // if(!this.alive) return;
        // console.log()
        this.dr = this.direction.x * Math.cos(this.phi) + this.direction.y * Math.sin(this.phi);
        this.dphi = (-this.direction.x * Math.sin(this.phi) + this.direction.y * Math.cos(this.phi)) / this.r;
        // this.dr += this.r * this.dphi * this.dphi - ((0.5 * this.blackHole.radius) / (this.r * this.r));
        // this.dphi += -2 * this.dr * this.dphi / this.r;
        // this.phi += this.dphi * STEP;
        this.r += this.dr * STEP;
        this.phi += this.dphi * STEP;
        // this.pos.add(this.direction.multiply(STEP, false));
        // if((blackHole.pos.x - this.pos.x ) * (blackHole.pos.x - this.pos.x) +
        //    (blackHole.pos.y - this.pos.y ) * (blackHole.pos.y - this.pos.y) < 
        //    (this.blackHole.radius + PHOTON_RADIUS) * (this.blackHole.radius + PHOTON_RADIUS)){
        //     this.alive = false;
        //     return;
        // }
        // this.prev_phi = this.phi;
        // this.prev_r = this.r;
        // this.phi += dphi_new;
        // this.r += ddr;
        this.pos.x = this.blackHole.pos.x + this.r * Math.cos(this.phi);
        this.pos.y = this.blackHole.pos.y + this.r * Math.sin(this.phi);
        // this.phi = Math.atan2(this.pos.y - this.blackHole.pos.y, this.pos.x - this.blackHole.pos.x);
        // this.r = Math.sqrt((this.pos.x - this.blackHole.pos.x) ** 2 + (this.pos.y - this.blackHole.pos.y) ** 2);
        if (this.r < this.blackHole.radius + PHOTON_RADIUS) {
            this.alive = false;
            return;
        }
    }
    draw() {
        if (!this.alive)
            return;
        ctx.beginPath();
        ctx.arc(this.pos.x, canvas.height - this.pos.y, PHOTON_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.closePath();
    }
}
class BlackHole {
    constructor(x, y, radius) {
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
let photons = [];
const blackHole = new BlackHole(canvas.width / 2, canvas.height / 2, 50);
const init = () => {
    for (let i = 0; i < numberOfPhotons; i++) {
        // const x = Math.random() * canvas.width;
        const x = 0;
        // const y = Math.random() * canvas.height;
        const y = (i + 0.5) * canvas.height / numberOfPhotons;
        const vx = 100 * (Math.random()) * 2;
        const vy = 0;
        // const vy = 100*(Math.random() - 0.5) * 2;
        photons.push(new Photon(x, y, vx, vy, blackHole));
    }
    mainLoop();
};
const mainLoop = () => {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    photons.forEach(photon => {
        photon.step();
        photon.draw();
    });
    blackHole.draw();
    requestAnimationFrame(mainLoop);
};
window.addEventListener("load", init);
