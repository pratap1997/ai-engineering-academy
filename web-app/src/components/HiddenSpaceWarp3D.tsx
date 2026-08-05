import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Layers, RotateCcw, Play, Pause, Sparkles } from 'lucide-react';

export const HiddenSpaceWarp3D: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [warpProgress, setWarpProgress] = useState<number>(0.8);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [showPlane, setShowPlane] = useState<boolean>(true);

  const warpRef = useRef<number>(warpProgress);
  const isPlayingRef = useRef<boolean>(isPlaying);

  useEffect(() => {
    warpRef.current = warpProgress;
  }, [warpProgress]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#090C10');

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(4, 3, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(6, 12, 0x10B981, 0x1F2937);
    gridHelper.position.y = -1;
    scene.add(gridHelper);

    // XOR Dataset points: (x, y) input space
    // (0,0) -> 0 (Red), (1,1) -> 0 (Red)
    // (0,1) -> 1 (Blue), (1,0) -> 1 (Blue)
    const pointsData = [
      { x: -1.2, y: -1.2, label: 0, color: 0xEF4444 }, // (0,0)
      { x: 1.2, y: 1.2, label: 0, color: 0xEF4444 },   // (1,1)
      { x: -1.2, y: 1.2, label: 1, color: 0x3B82F6 },  // (0,1)
      { x: 1.2, y: -1.2, label: 1, color: 0x3B82F6 },  // (1,0)
    ];

    const pointMeshes: THREE.Mesh[] = [];

    pointsData.forEach((p) => {
      const geometry = new THREE.SphereGeometry(0.18, 32, 32);
      const material = new THREE.MeshStandardMaterial({
        color: p.color,
        emissive: p.color,
        emissiveIntensity: 0.6,
        roughness: 0.2,
        metalness: 0.8,
      });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(p.x, -0.9, p.y);
      scene.add(sphere);
      pointMeshes.push(sphere);
    });

    // 3D Separating Hyperplane Mesh
    const planeGeo = new THREE.PlaneGeometry(3.5, 3.5);
    const planeMat = new THREE.MeshStandardMaterial({
      color: 0x10B981,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
      emissive: 0x10B981,
      emissiveIntensity: 0.2,
      wireframe: false,
    });
    const planeMesh = new THREE.Mesh(planeGeo, planeMat);
    planeMesh.rotation.x = Math.PI / 4;
    planeMesh.rotation.z = Math.PI / 6;
    planeMesh.position.set(0, 0.2, 0);
    scene.add(planeMesh);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x10B981, 1.5);
    dirLight.position.set(5, 8, 5);
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0x6366F1, 2, 10);
    pointLight.position.set(-3, 2, -3);
    scene.add(pointLight);

    // Animation & Orbit logic
    let angle = 0;
    let animId: number;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (isPlayingRef.current) {
        angle += 0.008;
      }

      // Smooth orbital camera movement
      camera.position.x = 4.5 * Math.cos(angle);
      camera.position.z = 4.5 * Math.sin(angle);
      camera.position.y = 2.5 + Math.sin(angle * 0.5) * 0.5;
      camera.lookAt(0, 0, 0);

      // Lift & Warp points in Z axis based on non-linear activation z = warp * x * y
      const currentWarp = warpRef.current;
      pointsData.forEach((p, idx) => {
        const targetZ = currentWarp * (p.x * p.y > 0 ? 1.2 : -1.2);
        pointMeshes[idx].position.y = targetZ;
      });

      // Hyperplane visibility
      planeMesh.visible = showPlane;

      renderer.render(scene, camera);
    };

    animate();

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
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [showPlane]);

  return (
    <div className="relative w-full h-[400px] bg-[#090C10] rounded-2xl border border-white/10 overflow-hidden shadow-2xl flex flex-col justify-between">
      {/* Top Header Overlay */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 bg-[#0D1117]/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 pointer-events-auto">
          <Layers className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-mono font-bold text-white">
            MLP 3D Hidden Space Warping
          </span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            Module 002
          </span>
        </div>

        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 bg-[#0D1117]/80 backdrop-blur-md hover:bg-emerald-500/20 border border-white/10 rounded-lg text-slate-300 hover:text-emerald-400 transition"
            title={isPlaying ? "Pause Camera Orbit" : "Play Camera Orbit"}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setShowPlane(!showPlane)}
            className={`px-3 py-1.5 text-xs font-mono rounded-lg border backdrop-blur-md transition ${
              showPlane
                ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                : 'bg-[#0D1117]/80 border-white/10 text-slate-400'
            }`}
          >
            {showPlane ? 'Hyperplane ON' : 'Hyperplane OFF'}
          </button>
        </div>
      </div>

      {/* 3D WebGL Canvas Mounting Container */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Bottom Interactive Controls */}
      <div className="absolute bottom-4 left-4 right-4 z-10 bg-[#0D1117]/90 backdrop-blur-md p-3.5 rounded-xl border border-white/10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3 flex-1 min-w-[200px]">
          <span className="text-xs font-mono text-slate-300 font-semibold whitespace-nowrap">
            Non-Linear Layer Warp: {(warpProgress * 100).toFixed(0)}%
          </span>
          <input
            type="range"
            min="0"
            max="1.5"
            step="0.05"
            value={warpProgress}
            onChange={(e) => setWarpProgress(parseFloat(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block shadow-[0_0_8px_#EF4444]" />
            <span className="text-slate-300">Class 0 (XOR 0,0 & 1,1)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block shadow-[0_0_8px_#3B82F6]" />
            <span className="text-slate-300">Class 1 (XOR 0,1 & 1,0)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
