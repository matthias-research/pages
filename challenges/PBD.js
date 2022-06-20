// The MIT License (MIT)
// Copyright (c) 2020 NVIDIA
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Position Based Dynamics Library
// Matthias Müller, NVIDIA

(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(global = global || self, factory(global.PBD = {}));
}(this, (function (exports) { 'use strict';


var maxRotationPerSubstep = 0.5;

// Pose  -----------------------------------------------------------

class Pose {
    constructor() {
        this.p = new THREE.Vector3(0.0, 0.0, 0.0);
        this.q = new THREE.Quaternion(0.0, 0.0, 0.0, 1.0);					
    }
    copy(pose) {
        this.p.copy(pose.p);
        this.q.copy(pose.q);
    }
    clone() {
        var newPose = new Pose();
        newPose.p = this.p.clone();
        newPose.q = this.q.clone();
        return newPose;
    }   
    rotate(v) {
        v.applyQuaternion(this.q);
    }
    invRotate(v) {
        let inv = this.q.clone();
        inv.conjugate();
        v.applyQuaternion(inv);
    }
    transform(v) {
        v.applyQuaternion(this.q);
        v.add(this.p);
    }
    invTransform(v) {
        v.sub(this.p);
        this.invRotate(v);
    }
    transformPose(pose) {
        pose.q.multiplyQuaternions(this.q, pose.q);
        this.rotate(pose.p);
        pose.p.add(this.p);
    }   
}

function getQuatAxis0(q) {
	let x2 = q.x * 2.0;
    let w2 = q.w * 2.0;
    return new THREE.Vector3((q.w * w2) - 1.0 + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2);
}
function getQuatAxis1(q) {
	let y2 = q.y * 2.0;
    let w2 = q.w * 2.0;
    return new THREE.Vector3((-q.z * w2) + q.x * y2, (q.w * w2) - 1.0 + q.y * y2, (q.x * w2) + q.z * y2);
}
function getQuatAxis2(q) {
	let z2 = q.z * 2.0;
	let w2 = q.w * 2.0;
	return new THREE.Vector3((q.y * w2) + q.x * z2, (-q.x * w2) + q.y * z2, (q.w * w2) - 1.0 + q.z * z2);
} 

// Rigid body class  -----------------------------------------------------------

class Body {
    constructor(pose, mesh) 
    { 
        this.pose = pose.clone();
        this.prevPose = pose.clone();
        this.origPose = pose.clone();
        this.vel = new THREE.Vector3(0.0, 0.0, 0.0);
        this.omega = new THREE.Vector3(0.0, 0.0, 0.0);
        
        this.invMass = 1.0;
        this.invInertia = new THREE.Vector3(1.0, 1.0, 1.0);
        this.mesh = mesh;
        this.mesh.position.copy(this.pose.p);
        this.mesh.quaternion.copy(this.pose.q);
        mesh.userData.physicsBody = this;
    }

    setBox(size, density = 1.0) {
        let mass = size.x * size.y * size.z * density;
        this.invMass = 1.0 / mass;
        mass /= 12.0;
        this.invInertia.set(
            1.0 / (size.y * size.y + size.z * size.z) / mass,
            1.0 / (size.z * size.z + size.x * size.x) / mass,
            1.0 / (size.x * size.x + size.y * size.y) / mass);
    }

    applyRotation(rot, scale = 1.0) {

        // safety clamping. This happens very rarely if the solver
        // wants to turn the body by more than 30 degrees in the
        // orders of milliseconds

        let maxPhi = 0.5;
        let phi = rot.length();
        if (phi * scale > maxRotationPerSubstep) 
            scale = maxRotationPerSubstep / phi;
            
        let dq = new THREE.Quaternion(rot.x * scale, rot.y * scale, rot.z * scale, 0.0);					
        dq.multiply(this.pose.q);
        this.pose.q.set(this.pose.q.x + 0.5 * dq.x, this.pose.q.y + 0.5 * dq.y, 
                this.pose.q.z + 0.5 * dq.z, this.pose.q.w + 0.5 * dq.w);
        this.pose.q.normalize();
    }

    integrate(dt, gravity) {
        this.prevPose.copy(this.pose);
        this.vel.addScaledVector(gravity, dt);					
        this.pose.p.addScaledVector(this.vel, dt);
        this.applyRotation(this.omega, dt);
    }

    update(dt) {
        this.vel.subVectors(this.pose.p, this.prevPose.p);
        this.vel.multiplyScalar(1.0 / dt);
        let dq = new THREE.Quaternion();
        dq.multiplyQuaternions(this.pose.q, this.prevPose.q.conjugate());
        this.omega.set(dq.x * 2.0 / dt, dq.y * 2.0 / dt, dq.z * 2.0 / dt);
        if (dq.w < 0.0)
            this.omega.set(-this.omega.x, -this.omega.y, -this.omega.z);

        // this.omega.multiplyScalar(1.0 - 1.0 * dt);
        // this.vel.multiplyScalar(1.0 - 1.0 * dt);

        this.mesh.position.copy(this.pose.p);
        this.mesh.quaternion.copy(this.pose.q);
    }

    getVelocityAt(pos) {					
        let vel = new THREE.Vector3(0.0, 0.0, 0.0);					
        vel.subVectors(pos, this.pose.p);
        vel.cross(this.omega);
        vel.subVectors(this.vel, vel);
        return vel;
    }

    getInverseMass(normal, pos = null) {
        let n = new THREE.Vector3();
        if (pos === null) 
            n.copy(normal);
        else {
            n.subVectors(pos, this.pose.p);
            n.cross(normal);
        }
        this.pose.invRotate(n);
        let w = 
            n.x * n.x * this.invInertia.x +
            n.y * n.y * this.invInertia.y +
            n.z * n.z * this.invInertia.z;
        if (pos !== null)
            w += this.invMass;
        return w;
    }

    applyCorrection(corr, pos = null, velocityLevel = false) {
        let dq = new THREE.Vector3();
        if (pos === null) 
            dq.copy(corr);
        else {
            if (velocityLevel)
                this.vel.addScaledVector(corr, this.invMass);
            else
                this.pose.p.addScaledVector(corr, this.invMass);
            dq.subVectors(pos, this.pose.p);
            dq.cross(corr);
        }
        this.pose.invRotate(dq);
        dq.set(this.invInertia.x * dq.x, 
            this.invInertia.y * dq.y, this.invInertia.z * dq.z);
        this.pose.rotate(dq);
        if (velocityLevel)
            this.omega.add(dq);
        else 
            this.applyRotation(dq);
    }
}

// ------------------------------------------------------------------------------------

function applyBodyPairCorrection(body0, body1, corr, compliance, dt, pos0 = null, pos1 = null, 
    velocityLevel = false) 
{
    let C = corr.length();
    if ( C == 0.0)
        return;

    let normal = corr.clone();
    normal.normalize();

    let w0 = body0 ? body0.getInverseMass(normal, pos0) : 0.0;
    let w1 = body1 ? body1.getInverseMass(normal, pos1) : 0.0;

    let w = w0 + w1;
    if (w == 0.0)
        return;

    let lambda = -C / (w + compliance / dt / dt);
    normal.multiplyScalar(-lambda);
    if (body0)
        body0.applyCorrection(normal, pos0, velocityLevel);
    if (body1) {
        normal.multiplyScalar(-1.0);
        body1.applyCorrection(normal, pos1, velocityLevel);
    }
}

// ------------------------------------------------------------------------------------------------

function limitAngle(body0, body1, n, a, b, minAngle, maxAngle, compliance, dt, maxCorr = Math.PI)
{
    // the key function to handle all angular joint limits
    let c = new THREE.Vector3();
    c.crossVectors(a, b);

    let phi = Math.asin(c.dot(n));
    if (a.dot(b) < 0.0)
        phi = Math.PI - phi;

    if (phi > Math.PI)
        phi -= 2.0 * Math.PI;
    if (phi < -Math.PI)
        phi += 2.0 * Math.PI;

    if (phi < minAngle || phi > maxAngle) {
        phi = Math.min(Math.max(minAngle, phi), maxAngle);

        let q = new THREE.Quaternion();
        q.setFromAxisAngle(n, phi);

        let omega = a.clone();
        omega.applyQuaternion(q);
        omega.cross(b);

        phi = omega.length();
        if (phi > maxCorr) 
            omega.multiplyScalar(maxCorr / phi);

        applyBodyPairCorrection(body0, body1, omega, compliance, dt);
    }
}	

// Joint class  -----------------------------------------------------------

const JointType = {
    SPHERICAL: "spherical",
    HINGE: "hinge",
    FIXED: "fixed"
}

class Joint {
    constructor(type, body0, body1, localPose0, localPose1) 
    { 
        this.body0 = body0;
        this.body1 = body1;
        this.localPose0 = localPose0.clone();
        this.localPose1 = localPose1.clone();
        this.globalPose0 = localPose0.clone();
        this.globalPose1 = localPose1.clone();

        this.type = type;					
        this.compliance = 0.0;
        this.rotDamping = 0.0;
        this.posDamping = 0.0;
        this.hasSwingLimits = false;
        this.minSwingAngle = -2.0 * Math.PI;
        this.maxSwingAngle = 2.0 * Math.PI;
        this.swingLimitsCompliance = 0.0;
        this.hasTwistLimits = false;
        this.minTwistAngle = -2.0 * Math.PI;
        this.maxTwistAngle = 2.0 * Math.PI;
        this.twistLimitCompliance = 0.0;
    }

    updateGlobalPoses() {
        this.globalPose0.copy(this.localPose0);
        if (this.body0)
            this.body0.pose.transformPose(this.globalPose0);
        this.globalPose1.copy(this.localPose1);
        if (this.body1)
            this.body1.pose.transformPose(this.globalPose1);
    }

    solvePos(dt) {

        this.updateGlobalPoses();

        // orientation

        if (this.type == JointType.FIXED) {
            let q = this.globalPose0.q;
            q.conjugate();
            q.multiplyQuaternions(this.globalPose1.q, q);
            let omega = new THREE.Vector3();
            omega.set(2.0 * q.x, 2.0 * q.y, 2.0 * q.z);
            if (omega.w < 0.0)
                omega.multiplyScalar(-1.0);
            applyBodyPairCorrection(this.body0, this.body1, omega, this.compliance, dt);						
        }

        if (this.type == JointType.HINGE) {

            // align axes
            let a0 = getQuatAxis0(this.globalPose0.q);
            let b0 = getQuatAxis1(this.globalPose0.q);
            let c0 = getQuatAxis2(this.globalPose0.q);
            let a1 = getQuatAxis0(this.globalPose1.q);
            a0.cross(a1);
            applyBodyPairCorrection(this.body0, this.body1, a0, 0.0, dt);

            // limits
            if (this.hasSwingLimits) {
                this.updateGlobalPoses();
                let n = getQuatAxis0(this.globalPose0.q);
                let b0 = getQuatAxis1(this.globalPose0.q);
                let b1 = getQuatAxis1(this.globalPose1.q);
                limitAngle(this.body0, this.body1, n, b0, b1, 
                    this.minSwingAngle, this.maxSwingAngle, this.swingLimitsCompliance, dt);
            }
        }

        if (this.type == JointType.SPHERICAL) {

            // swing limits
            if (this.hasSwingLimits) {
                this.updateGlobalPoses();
                let a0 = getQuatAxis0(this.globalPose0.q);
                let a1 = getQuatAxis0(this.globalPose1.q);
                let n = new THREE.Vector3();
                n.crossVectors(a0, a1);
                n.normalize();
                limitAngle(this.body0, this.body1, n, a0, a1, 
                    this.minSwingAngle, this.maxSwingAngle, this.swingLimitsCompliance, dt);
            }
            // twist limits
            if (this.hasTwistLimits) {
                this.updateGlobalPoses();
                let n0 = getQuatAxis0(this.globalPose0.q);
                let n1 = getQuatAxis0(this.globalPose1.q);
                let n = new THREE.Vector3();
                n.addVectors(n0, n1)
                n.normalize();
                let a0 = getQuatAxis1(this.globalPose0.q);
                a0.addScaledVector(n, -n.dot(a0));
                a0.normalize();
                let a1 = getQuatAxis1(this.globalPose1.q);
                a1.addScaledVector(n, -n.dot(a1));
                a1.normalize();

                // handling gimbal lock problem
                let maxCorr = n0.dot(n1) > -0.5 ? 2.0 * Math.Pi : 1.0 * dt;		
               
                limitAngle(this.body0, this.body1, n, a0, a1, 
                    this.minTwistAngle, this.maxTwistAngle, this.twistLimitCompliance, dt, maxCorr);
            }
        }

        // position
        
        // simple attachment

        this.updateGlobalPoses();
        let corr = new THREE.Vector3();
        corr.subVectors(this.globalPose1.p, this.globalPose0.p);
        applyBodyPairCorrection(this.body0, this.body1, corr, this.compliance, dt,
            this.globalPose0.p, this.globalPose1.p);	
    }

    solveVel(dt) { 

        // Gauss-Seidel lets us make damping unconditionally stable in a 
        // very simple way. We clamp the correction for each constraint
        // to the magnitude of the currect velocity making sure that
        // we never subtract more than there actually is.

        if (this.rotDamping > 0.0) {
            let omega = new THREE.Vector3(0.0, 0.0, 0.0);
            if (this.body0)
                omega.sub(this.body0.omega);
            if (this.body1)
                omega.add(this.body1.omega); 
            omega.multiplyScalar(Math.min(1.00, this.rotDamping * dt));
            applyBodyPairCorrection(this.body0, this.body1, omega, 0.0, dt, 
                    null, null, true);
        }
        if (this.posDamping > 0.0) {
            this.updateGlobalPoses();
            let vel = new THREE.Vector3(0.0, 0.0, 0.0);
            if (this.body0)
                vel.sub(this.body0.getVelocityAt(this.globalPose0.p));
            if (this.body1)
                vel.add(this.body1.getVelocityAt(this.globalPose1.p));
            vel.multiplyScalar(Math.min(1.0, this.posDamping * dt));
            applyBodyPairCorrection(this.body0, this.body1, vel, 0.0, dt, 
                    this.globalPose0.p, this.globalPose1.p, true);
        }
    }	
}

// Simulate -----------------------------------------------------------

function simulate(bodies, joints, timeStep, numSubsteps, gravity) {
    let dt = timeStep / numSubsteps;

    for (let i = 0; i < numSubsteps; i++) {
        for (let j = 0; j < bodies.length; j++) 
            bodies[j].integrate(dt, gravity);

        for (let j = 0; j < joints.length; j++)
            joints[j].solvePos(dt);

        for (let j = 0; j < bodies.length; j++) 
            bodies[j].update(dt);

        for (let j = 0; j < joints.length; j++)
            joints[j].solveVel(dt);
    }
}

exports.Pose = Pose;
exports.Body = Body;
exports.JointType = JointType;
exports.Joint = Joint;

exports.simulate = simulate;

Object.defineProperty(exports, '__esModule', { value: true });

})));
