import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Globe, Radio, Zap, ShieldCheck } from 'lucide-react';

export const GlobalSwarmGlobe3D: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [activeNodesCount, setActiveNodesCount] = useState<number>(7);
  const [latencyMs, setLatencyMs] = useState<number>(42);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);

  const isPlayingRef = useRef<boolean>(isPlaying);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#090C10');

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 3D Wireframe Globe
    const globeRadius = 2.0;
    const globeGeo = new THREE.SphereGeometry(globeRadius, 36, 36);

    // Wireframe Mesh for Globe
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: 0x10B981,
      wireframe: true,
      transparent: true,
      opacity: 0.15,
    });
    const globeWireframe = new THREE.Mesh(globeGeo, wireframeMat);
    scene.add(globeWireframe);

    // Dark Solid Core Globe
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x0D1117,
      roughness: 0.9,
      metalness: 0.1,
    });
    const globeCore = new THREE.Mesh(globeGeo, coreMat);
    scene.add(globeCore);

    // Global Agent Nodes Coordinates (Lat, Lon)
    const nodes = [
      { name: 'San Francisco', lat: 37.7749, lon: -122.4194, color: 0x10B981 },
      { name: 'London', lat: 51.5074, lon: -0.1278, color: 0x3B82F6 },
      { name: 'Tokyo', lat: 35.6762, lon: 139.6503, color: 0xF59E0B },
      { name: 'Frankfurt', lat: 50.1109, lon: 8.6821, color: 0x8B5CF6 },
      { name: 'Sydney', lat: -33.8688, lon: 151.2093, color: 0xEC4899 },
      { name: 'Bengaluru', lat: 12.9716, lon: 77.5946, color: 0x06B6D4 },
      { name: 'São Paulo', lat: -23.5505, lon: -46.6333, color: 0x10B981 },
    ];

    // Convert Lat/Lon to 3D Cartesian coordinates
    const latLonToVector3 = (lat: number, lon: number, radius: number) => {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = (lon + 180) * (Math.PI / 180);

      const x = -(radius * Math.sin(phi) * Math.cos(theta));
      const z = radius * Math.sin(phi) * Math.sin(theta);
      const y = radius * Math.cos(phi);

      return new THREE.Vector3(x, y, z);
    };

    // Node Meshes & Glowing Rings
    const nodeVectorGroup = new THREE.Group();

    nodes.forEach((n) => {
      const pos = latLonToVector3(n.lat, n.lon, globeRadius + 0.05);

      // Sphere Marker
      const nodeGeo = new THREE.SphereGeometry(0.06, 16, 16);
      const nodeMat = new THREE.MeshBasicMaterial({ color: n.color });
      const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
      nodeMesh.position.copy(pos);
      nodeVectorGroup.add(nodeMesh);

      // Outer Pulse Ring
      const ringGeo = new THREE.RingGeometry(0.08, 0.12, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: n.color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.position.copy(pos.clone().multiplyScalar(1.02));
      ringMesh.lookAt(0, 0, 0);
      nodeVectorGroup.add(ringMesh);
    });

    scene.add(nodeVectorGroup);

    // Glowing Communication Arcs between nodes
    const arcMaterial = new THREE.LineBasicMaterial({
      color: 0x10B981,
      transparent: true,
      opacity: 0.5,
    });

    for (let i = 0; i < nodes.length; i++) {
      const start = latLonToVector3(nodes[i].lat, nodes[i].lon, globeRadius + 0.05);
      const nextIdx = (i + 1) % nodes.length;
      const end = latLonToVector3(nodes[nextIdx].lat, nodes[nextIdx].lon, globeRadius + 0.05);

      // Curve mid-point lifted above globe surface
      const mid = start.clone().add(end).multiplyScalar(0.5);
      mid.normalize().multiplyScalar(globeRadius + 0.8);

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const points = curve.getPoints(50);
      const arcGeo = new THREE.BufferGeometry().setFromPoints(points);

      const arcLine = new THREE.Line(arcGeo, arcMaterial);
      nodeVectorGroup.add(arcLine);
    }

    // Ambient & Point Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x10B981, 1.2);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);

    // Animation Loop
    let angle = 0;
    let animId: number;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (isPlayingRef.current) {
        nodeVectorGroup.rotation.y += 0.005;
        globeWireframe.rotation.y += 0.005;
        globeCore.rotation.y += 0.005;
      }

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
  }, []);

  return (
    <div className="relative w-full h-[420px] bg-[#090C10] rounded-2xl border border-white/10 overflow-hidden shadow-2xl flex flex-col justify-between my-4">
      {/* Top Overlay Header */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 bg-[#0D1117]/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 pointer-events-auto">
          <Globe className="w-4 h-4 text-emerald-400 animate-pulse" />
          <span className="text-xs font-mono font-bold text-white">
            Distributed Swarm P2P Mesh
          </span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            Module 048 / 049
          </span>
        </div>

        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="flex items-center gap-1.5 bg-[#0D1117]/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-xs font-mono text-emerald-400">
            <Radio className="w-3.5 h-3.5 text-emerald-400" />
            <span>P2P Consensus Active</span>
          </div>
        </div>
      </div>

      {/* 3D WebGL Canvas */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Bottom Telemetry Bar */}
      <div className="absolute bottom-4 left-4 right-4 z-10 bg-[#0D1117]/90 backdrop-blur-md p-3.5 rounded-xl border border-white/10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="text-slate-300">Nodes Online: <strong className="text-emerald-400">7 Regions</strong></span>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-slate-300">Mesh Latency: <strong className="text-yellow-400">{latencyMs}ms</strong></span>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
          <span>SF • London • Tokyo • Frankfurt • Sydney • Bengaluru • SP</span>
        </div>
      </div>
    </div>
  );
};
