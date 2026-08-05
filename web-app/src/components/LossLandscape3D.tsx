import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface LossLandscape3DProps {
  learningRate?: number;
  isTraining?: boolean;
}

/**
 * LossLandscape3D — 60fps 3D WebGL Loss Surface Visualizer built with Three.js.
 * Renders a 3D convex loss bowl z = x^2 + y^2 with a ball rolling down the gradient trajectory.
 */
export const LossLandscape3D: React.FC<LossLandscape3DProps> = ({
  learningRate = 0.05,
  isTraining = true,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090c10);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(12, 14, 18);
    camera.lookAt(0, 0, 0);

    // 2. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 3. Grid & Ambient Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x10b981, 1.5);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    // 4. 3D Loss Bowl Surface Geometry (z = 0.15 * (x^2 + y^2))
    const gridSize = 40;
    const segments = 60;
    const geometry = new THREE.PlaneGeometry(gridSize, gridSize, segments, segments);
    geometry.rotateX(-Math.PI / 2);

    const posAttr = geometry.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const z = posAttr.getZ(i);
      const y = 0.12 * (x * x + z * z);
      posAttr.setY(i, y);
    }
    geometry.computeVertexNormals();

    const wireframeMaterial = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
    });
    const surfaceMesh = new THREE.Mesh(geometry, wireframeMaterial);
    scene.add(surfaceMesh);

    // Solid inner bowl with gradient color
    const solidMaterial = new THREE.MeshPhongMaterial({
      color: 0x090c10,
      emissive: 0x064e3b,
      side: THREE.DoubleSide,
      shininess: 40,
    });
    const solidMesh = new THREE.Mesh(geometry, solidMaterial);
    scene.add(solidMesh);

    // 5. Gradient Descent Ball (Neuron Weight State)
    const ballGeo = new THREE.SphereGeometry(0.6, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x10b981,
      emissiveIntensity: 0.8,
      roughness: 0.2,
    });
    const ballMesh = new THREE.Mesh(ballGeo, ballMat);
    scene.add(ballMesh);

    // Trajectory Line
    const maxTrailPoints = 100;
    const trailPositions = new Float32Array(maxTrailPoints * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    const trailMat = new THREE.LineBasicMaterial({ color: 0x34d399, linewidth: 3 });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    scene.add(trailLine);

    // 6. Simulation State
    let ballX = 14.0;
    let ballZ = 12.0;
    let trailCount = 0;

    const updateBall = () => {
      if (!isTraining) return;

      // Gradient of z = 0.12 * (x^2 + z^2) -> dz/dx = 0.24*x, dz/dz = 0.24*z
      const gradX = 0.24 * ballX;
      const gradZ = 0.24 * ballZ;

      ballX -= learningRate * gradX;
      ballZ -= learningRate * gradZ;

      const ballY = 0.12 * (ballX * ballX + ballZ * ballZ) + 0.6;
      ballMesh.position.set(ballX, ballY, ballZ);

      // Record trajectory trail
      if (trailCount < maxTrailPoints) {
        const positions = trailGeo.attributes.position.array as Float32Array;
        positions[trailCount * 3] = ballX;
        positions[trailCount * 3 + 1] = ballY;
        positions[trailCount * 3 + 2] = ballZ;
        trailGeo.attributes.position.needsUpdate = true;
        trailCount++;
        trailGeo.setDrawRange(0, trailCount);
      }

      // Reset when close to local minimum
      if (Math.abs(ballX) < 0.1 && Math.abs(ballZ) < 0.1) {
        ballX = (Math.random() > 0.5 ? 1 : -1) * (10 + Math.random() * 5);
        ballZ = (Math.random() > 0.5 ? 1 : -1) * (10 + Math.random() * 5);
        trailCount = 0;
        trailGeo.setDrawRange(0, 0);
      }
    };

    // 7. Animation Loop
    let animationFrameId: number;
    let angle = 0;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // Rotate camera around origin slowly for cinematic 3D perspective
      angle += 0.005;
      camera.position.x = 22 * Math.sin(angle);
      camera.position.z = 22 * Math.cos(angle);
      camera.lookAt(0, 3, 0);

      updateBall();
      renderer.render(scene, camera);
    };

    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [learningRate, isTraining]);

  return (
    <div
      ref={mountRef}
      style={{
        width: '100%',
        height: '420px',
        backgroundColor: '#090C10',
        borderRadius: '16px',
        border: '1px solid rgba(16, 185, 129, 0.3)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* 3D HUD Badge Overlay */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          border: '1px solid #10B981',
          borderRadius: '20px',
          padding: '6px 14px',
          fontSize: '13px',
          fontFamily: 'monospace',
          color: '#10B981',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          zIndex: 10,
          backdropFilter: 'blur(6px)',
        }}
      >
        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#10B981', animation: 'pulse 1.5s infinite' }} />
        3D WEBGL LOSS SURFACE — z = f(w1, w2)
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          fontSize: '12px',
          fontFamily: 'sans-serif',
          color: '#94A3B8',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
          padding: '4px 10px',
          borderRadius: '6px',
          zIndex: 10,
        }}
      >
        Interactive 60fps Gradient Trajectory
      </div>
    </div>
  );
};
