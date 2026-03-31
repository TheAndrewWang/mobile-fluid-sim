<script lang="ts">
	import Loader2 from '@lucide/svelte/icons/loader-2';
	import Smartphone from '@lucide/svelte/icons/smartphone';

	import { onMount, onDestroy } from 'svelte';
	import { Tween } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';

	import { browser } from '$app/environment';
	import FluidSimulation from '$lib/FluidSimulation.svelte';
	import GitHubLink from '$lib/GitHubLink.svelte';
	import PopupInfo from '$lib/PopupInfo.svelte';

	const GRAVITY_SCALE = 50.81;
	const SHAKE_THRESHOLD = 18;
	const SHAKE_COOLDOWN = 1000;
	const SHAKE_SMOOTHING = 0.15;
	const SHAKE_IMPULSE_STRENGTH = -80;
	const SHAKE_IMPULSE_DECAY = 0.9;

	let rawX = 0;
	let rawY = 0;
	let shakeImpulseY = 0;

	type AppState = 'loading' | 'needs-permission' | 'ready' | 'denied' | 'not-supported';

	const fluidTypes = [
		{
			fluidColor: { r: 0.09, g: 0.4, b: 1.0 },
			foamColor: { r: 0.09, g: 0.4, b: 1.0 },
			colorDiffusionCoeff: 0.0008,
			foamReturnRate: 0.5
		},
		{
			fluidColor: { r: 0.0, g: 0.7, b: 0.8 },
			foamColor: { r: 0.0, g: 0.7, b: 0.8 },
			colorDiffusionCoeff: 0.0012,
			foamReturnRate: 0.6
		},
		{
			fluidColor: { r: 1.0, g: 0.4, b: 0.1 },
			foamColor: { r: 1.0, g: 0.4, b: 0.1 },
			colorDiffusionCoeff: 0.0004,
			foamReturnRate: 0.3
		},
		{
			fluidColor: { r: 0.5, g: 0.2, b: 0.9 },
			foamColor: { r: 0.5, g: 0.2, b: 0.9 },
			colorDiffusionCoeff: 0.001,
			foamReturnRate: 0.7
		},
		{
			fluidColor: { r: 0.1, g: 0.6, b: 0.4 },
			foamColor: { r: 0.1, g: 0.6, b: 0.4 },
			colorDiffusionCoeff: 0.0015,
			foamReturnRate: 0.4
		},
		{
			fluidColor: { r: 0.9, g: 0.5, b: 0.6 },
			foamColor: { r: 0.9, g: 0.5, b: 0.6 },
			colorDiffusionCoeff: 0.0006,
			foamReturnRate: 0.8
		},
		{
			fluidColor: { r: 0.3, g: 0.7, b: 0.9 },
			foamColor: { r: 0.3, g: 0.7, b: 0.9 },
			colorDiffusionCoeff: 0.0009,
			foamReturnRate: 0.5
		},
		{
			fluidColor: { r: 0.9, g: 0.7, b: 0.2 },
			foamColor: { r: 0.9, g: 0.7, b: 0.2 },
			colorDiffusionCoeff: 0.0005,
			foamReturnRate: 0.2
		}
	];

	let currentFluidIndex = $state(0);

	let angle: number | undefined = $state(0);
	let gravity: { x: number; y: number } = $state({ x: 0, y: -GRAVITY_SCALE });
	let fluidColor = new Tween(fluidTypes[0].fluidColor, {
		duration: 500,
		easing: cubicOut
	});
	let foamColor = new Tween(fluidTypes[0].foamColor, {
		duration: 500,
		easing: cubicOut
	});
	let colorDiffusionCoeff: number = $state(fluidTypes[0].colorDiffusionCoeff);
	let foamReturnRate: number = $state(fluidTypes[0].foamReturnRate);

	let appState: AppState = $state('loading');

	// Shake detection
	let lastShakeTime = 0;
	let lastMotionTime = 0;
	let filteredShake = 0;

	const requestPermission = async () => {
		if (!browser) return;

		if (
			'DeviceOrientationEvent' in window &&
			typeof (DeviceOrientationEvent as any).requestPermission === 'function'
		) {
			try {
				const orientationResponse = await (DeviceOrientationEvent as any).requestPermission();
				let motionResponse = 'granted';

				if (
					'DeviceMotionEvent' in window &&
					typeof (DeviceMotionEvent as any).requestPermission === 'function'
				) {
					motionResponse = await (DeviceMotionEvent as any).requestPermission();
				}

				if (orientationResponse === 'granted' && motionResponse === 'granted') {
					startListening();
					appState = 'ready';
				} else {
					appState = 'denied';
				}
			} catch (error) {
				console.error('Error requesting device motion/orientation permission:', error);
				appState = 'denied';
			}
		} else if ('DeviceOrientationEvent' in window) {
			startListening();
			appState = 'ready';
		} else {
			appState = 'not-supported';
		}
	};

	onMount(() => {
		window.addEventListener('click', onTap);
		return () => {
			window.removeEventListener('click', onTap);
		};
	});

	const startListening = () => {
		if (!browser) return;
		window.addEventListener('deviceorientation', onOrientationChange);
		window.addEventListener('devicemotion', onDeviceMotion);
	};

	const onDeviceMotion = (event: DeviceMotionEvent) => {
		const now = performance.now();
		const dt = lastMotionTime ? Math.max((now - lastMotionTime) / 1000, 1 / 120) : 1 / 60;
		lastMotionTime = now;

		const acceleration = event.acceleration ?? event.accelerationIncludingGravity;
		if (!acceleration) return;

		const ax = acceleration.x ?? 0;
		const ay = acceleration.y ?? 0;
		const az = acceleration.z ?? 0;

		const magnitude = Math.sqrt(ax * ax + ay * ay + az * az);

		// Normalize by time so different sensor event rates feel more consistent.
		const normalizedMagnitude = magnitude / Math.sqrt(dt);

		// Smooth out noisy sensor spikes across devices.
		filteredShake =
			filteredShake * (1 - SHAKE_SMOOTHING) + normalizedMagnitude * SHAKE_SMOOTHING;

		const currentTime = Date.now();
		if (filteredShake > SHAKE_THRESHOLD && currentTime - lastShakeTime > SHAKE_COOLDOWN) {
			onShake();
			lastShakeTime = currentTime;
			shakeImpulseY = SHAKE_IMPULSE_STRENGTH;
		}
	};

	const onOrientationChange = (event: DeviceOrientationEvent) => {
		if (event.beta !== null && event.gamma !== null) {
			const beta = event.beta;
			const gamma = event.gamma;

			const betaRad = beta * (Math.PI / 180);
			const gammaRad = gamma * (Math.PI / 180);

			const cosBeta = Math.cos(betaRad);
			const sinBeta = Math.sin(betaRad);
			const sinGamma = Math.sin(gammaRad);

			const gx = sinGamma * cosBeta;
			const gy = -sinBeta;

			rawX = GRAVITY_SCALE * Math.max(-1, Math.min(1, gx));
			rawY = GRAVITY_SCALE * Math.max(-1, Math.min(1, gy));
			angle = gamma;
		}
	};

	let updateLoopFrame = 0;
	function updateLoop() {
		shakeImpulseY *= SHAKE_IMPULSE_DECAY;

		gravity.x = rawX;
		gravity.y = rawY + shakeImpulseY;

		updateLoopFrame = requestAnimationFrame(updateLoop);
	}

	onMount(() => {
		updateLoopFrame = requestAnimationFrame(updateLoop);
		return () => cancelAnimationFrame(updateLoopFrame);
	});

	onMount(async () => {
		if (!browser) return;

		if (!('DeviceOrientationEvent' in window)) {
			gravity = { x: 0, y: -GRAVITY_SCALE };
			angle = 0;
			appState = 'not-supported';
			return;
		}

		if (typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
			appState = 'needs-permission';
		} else {
			startListening();
			appState = 'ready';
		}
	});

	let isMouseDown = $state(false); // Track the button state

	const onMouseDown = () => { isMouseDown = true; };
	const onMouseUp = () => { isMouseDown = false; };

	const onMouseMove = (event: MouseEvent) => {
		// Only calculate "tilt" if we are on a desktop AND the button is held
		if ((appState === 'not-supported' || appState === 'ready') && isMouseDown) {
			const x = (event.clientX / window.innerWidth) - 0.5;
			const y = (event.clientY / window.innerHeight) - 0.5;
			
			// Multiply by 2 to give it a stronger "tilt" feel
			rawX = x * GRAVITY_SCALE * 2;
			rawY = -y * GRAVITY_SCALE * 2;
		}
	};

	onMount(() => {
		// Standard listeners for the mouse movement
		window.addEventListener('mousemove', onMouseMove);
		window.addEventListener('mousedown', onMouseDown);
		window.addEventListener('mouseup', onMouseUp);

		return () => {
			window.removeEventListener('mousemove', onMouseMove);
			window.removeEventListener('mousedown', onMouseDown);
			window.removeEventListener('mouseup', onMouseUp);
		};
	});

	onDestroy(() => {
		if (browser && window) {
			window.removeEventListener('deviceorientation', onOrientationChange);
			window.removeEventListener('devicemotion', onDeviceMotion);
		}
	});

	const onShake = () => {
		currentFluidIndex = (currentFluidIndex + 1) % fluidTypes.length;
		const newFluid = fluidTypes[currentFluidIndex];

		fluidColor.target = newFluid.fluidColor;
		foamColor.target = newFluid.foamColor;
		colorDiffusionCoeff = newFluid.colorDiffusionCoeff;
		foamReturnRate = newFluid.foamReturnRate;
	};

	const onTap = () => {
		currentFluidIndex = (currentFluidIndex + 1) % fluidTypes.length;
		const newFluid = fluidTypes[currentFluidIndex];

		fluidColor.target = newFluid.fluidColor;
		foamColor.target = newFluid.foamColor;
		colorDiffusionCoeff = newFluid.colorDiffusionCoeff;
		foamReturnRate = newFluid.foamReturnRate;
	};

	let isMenuOpen = $state(false);

	const toggleMenu = (event: MouseEvent) => {
		// stopPropagation prevents the "onTap" fluid color change 
		// from firing at the same time as the menu opening
		event.stopPropagation(); 
		isMenuOpen = !isMenuOpen;
	};

	const selectElement = (elementName: string, elementPNG: string) => {
		// Logic to change fluid behavior based on element
		console.log("Selected:", elementName);
		isMenuOpen = false;
		
	};
</script>

<div
	class="relative flex h-screen flex-col items-center justify-center bg-gradient-to-b from-gray-900 via-gray-800 to-gray-950"
>
	
	<button 
		onclick={toggleMenu} 
		class="absolute top-8 left-8 z-50 focus:outline-none transition-transform hover:scale-110 active:scale-95 text-white"
		aria-label="Open menu"
		title="Open menu"
		aria-expanded={isMenuOpen}
	>
		<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/>
		</svg>
	</button>

{#if isMenuOpen}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div 
        class="fixed inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-xl"
        role="button"
        tabindex="0"
        aria-label="Close menu"
        onclick={() => isMenuOpen = false}
    >
        <div 
            class="relative max-w-5xl w-[95vw] rounded-3xl bg-gray-900/90 p-8 shadow-2xl border border-gray-700/50"
            role="dialog"
            aria-modal="true"
            aria-label="Periodic Table Element Selection"
            tabindex="0"
            onclick={(e) => e.stopPropagation()} 
        >
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-black tracking-tighter text-white uppercase">Select Element</h2>
                <button onclick={() => isMenuOpen = false} class="text-gray-500 hover:text-white transition-colors">
                    ✕ Close
                </button>
            </div>

            <div class="grid grid-cols-18 gap-1 md:gap-2 overflow-x-auto pb-4">
                
                <button class="element-slot group" title="Hydrogen">
                    <img src="h.png" alt="H" class="element-img" />
                </button>
                
                <div class="col-span-16"></div> 
                
                <button class="element-slot group" title="Helium">
                    <img src="he.png" alt="He" class="element-img" />
                </button>

                <button class="element-slot group" title="Lithium">
                    <img src="li.png" alt="Li" class="element-img" />
                </button>
                <button class="element-slot group" title="Beryllium">
                    <img src="be.png" alt="Be" class="element-img" />
                </button>
                
                <div class="col-span-10"></div>

                <button class="element-slot group" title="Boron">
                    <img src="b.png" alt="B" class="element-img" />
                </button>
                <button onclick={() => selectElement('Carbon', 'carbon.png')} class="element-slot group" title="Carbon">
                    <img src="carbon.png" alt="C" class="element-img" />
                </button>
                </div>
            
            <p class="mt-4 text-center text-xs text-gray-500 italic">
                Click an element to inject it into the fluid simulation.
            </p>
        </div>
    </div>
{/if}

	<GitHubLink />
	{#if appState === 'loading'}
		<div class="text-center">
			<Loader2 class="mx-auto mb-4 h-12 w-12 animate-spin text-blue-400" />
			<h1 class="mb-2 text-2xl font-bold text-white">Fluid Simulation</h1>
			<p class="text-gray-300">Initializing...</p>
		</div>
	{:else if appState === 'needs-permission'}
		<div class="text-center">
			<Smartphone class="mx-auto mb-6 h-16 w-16 text-blue-400" />
			<h1 class="mb-4 text-2xl font-bold text-white">Motion Sensors Required</h1>
			<p class="mb-6 max-w-sm text-gray-300">
				This fluid simulation responds to your device's tilt and orientation. It also detects shake
				gestures to change fluid colors! Please grant permission to access motion sensors for the
				best experience.
			</p>
			<button
				onclick={requestPermission}
				class="rounded-lg bg-blue-500 px-8 py-3 font-semibold text-white shadow-lg transition-colors hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 focus:outline-none"
			>
				Enable Motion Sensors
			</button>
		</div>
	{:else if appState === 'denied'}
		<div class="text-center">
			<div class="mb-6 text-6xl">🚫</div>
			<h2 class="mb-4 text-xl font-semibold text-white">Permission Denied</h2>
			<p class="max-w-sm text-center text-gray-300">
				Motion sensor access was denied. You can still use the simulation, but it won't respond to
				device tilt. To enable this feature, please allow motion access in your browser settings.
			</p>
			<button
				onclick={requestPermission}
				class="mt-4 rounded-lg bg-gray-600 px-6 py-2 text-white transition-colors hover:bg-gray-500"
			>
				Try Again
			</button>
		</div>
	{:else}
		<FluidSimulation
			{gravity}
			fluidColor={fluidColor.current}
			foamColor={foamColor.current}
			{colorDiffusionCoeff}
			{foamReturnRate}
		/>

		{#if appState === 'ready'}
			<PopupInfo />
		{/if}
	{/if}
</div>

