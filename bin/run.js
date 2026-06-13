#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const repoRoot = path.join(__dirname, '..');
const buildDir = path.join(repoRoot, 'build');

console.log("==========================================");
console.log(" CUDA-AES Benchmark Runner");
console.log("==========================================");

try {
    if (!fs.existsSync(buildDir)) {
        console.log("=> Creating build directory...");
        fs.mkdirSync(buildDir);
    }
    
    console.log("=> Configuring with CMake...");
    execSync('cmake ..', { cwd: buildDir, stdio: 'inherit' });
    
    console.log("=> Compiling CUDA sources...");
    execSync('cmake --build . --config Release', { cwd: buildDir, stdio: 'inherit' });
    
    console.log("=> Running Benchmark...");
    let exePath = path.join(buildDir, 'Release', 'CudaProject.exe');
    if (!fs.existsSync(exePath)) {
        exePath = path.join(buildDir, 'CudaProject');
    }
    
    const args = process.argv.slice(2).join(' ');
    execSync(`"${exePath}" ${args}`, { cwd: repoRoot, stdio: 'inherit' });
    
} catch (err) {
    console.error("Error executing benchmark. Ensure CMake, MSVC, and CUDA Toolkit are installed.");
    process.exit(1);
}
