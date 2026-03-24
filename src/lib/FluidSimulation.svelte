<script lang="ts">
	import { onMount } from 'svelte';

	import { setupFluidScene, FluidRenderer } from '$lib/fluid';
	import type { FlipFluid } from '$lib/fluid';

	type QualityPreset = {
		resolution: number;
		numPressureIters: number;
		numParticleIters: number;
		damping: number;
		maxDpr: number;
	};

	let {
		gravity = { x: 0, y: -9.81 },
		resolution = 70,
		fluidColor = { r: 0.09, g: 0.4, b: 1.0 },
		foamColor = { r: 0.75, g: 0.9, b: 1.0 },
		colorDiffusionCoeff = 0.0008,
		foamReturnRate = 0.5,
		onclick
	}: {
		gravity?: { x: number; y: number };
		resolution?: number;
		angle?: number;
		fluidColor?: { r: number; g: number; b: number };
		foamColor?: { r: number; g: number; b: number };
		colorDiffusionCoeff?: number;
		foamReturnRate?: number;
		onclick?: () => void;
	} = $props();

	let canvas: HTMLCanvasElement;
	let fluid: FlipFluid;
	let renderer: FluidRenderer;
	let animationId: number;

	let simHeight = 3.0;
	let simWidth = 4.0;

	const dt = 1.0 / 90.0;
	const flipRatio = 0.95;
	const overRelaxation = 1.7;
	const compensateDrift = true;
	const separateParticles = true;
	const showParticles = false; // set true to overlay raw particles on top
	const showFluid = true; // metaball fluid surface
	const showGrid = false;

	let numPressureIters = 60;
	let numParticleIters = 3;
	let damping = 0.95;
	let effectiveResolution = resolution;
	let maxDpr = 2;
	let isPageVisible = true;

	// Particle count controls
	const relWaterWidth = 0.6; // Water width as fraction of tank (0.1 to 1.0)
	const relWaterHeight = 0.8; // Water height as fraction of tank (0.1 to 1.0)

	function isLikelyMobileDevice() {
		if (typeof window === 'undefined') return false;
		const hasTouch = navigator.maxTouchPoints > 0;
		const smallScreen = Math.min(window.innerWidth, window.innerHeight) <= 900;
		return hasTouch && smallScreen;
	}

	function getQualityPreset(): QualityPreset {
		const cpuThreads = navigator.hardwareConcurrency ?? 4;
		const mobile = isLikelyMobileDevice();

		if (mobile && cpuThreads <= 4) {
			return {
				resolution: Math.min(resolution, 46),
				numPressureIters: 22,
				numParticleIters: 2,
				damping: 0.965,
				maxDpr: 1.1
			};
		}

		if (mobile) {
			return {
				resolution: Math.min(resolution, 54),
				numPressureIters: 28,
				numParticleIters: 2,
				damping: 0.96,
				maxDpr: 1.25
			};
		}

		return {
			resolution,
			numPressureIters: 60,
			numParticleIters: 3,
			damping: 0.95,
			maxDpr: 1.75
		};
	}

	function resizeCanvas() {
		if (!canvas) return;

		const rect = canvas.getBoundingClientRect();
		const devicePixelRatio = window.devicePixelRatio || 1;
		const renderPixelRatio = Math.min(devicePixelRatio, maxDpr);

		canvas.width = rect.width * renderPixelRatio;
		canvas.height = rect.height * renderPixelRatio;

		// Update simulation dimensions to maintain aspect ratio
		const cScale = canvas.height / simHeight;
		simWidth = canvas.width / cScale;

		if (renderer) {
			renderer.resize(canvas.width, canvas.height);
		}
	}

	function simulate() {
		if (!fluid) return;

		fluid.simulate(
			dt,
			gravity.x,
			gravity.y,
			flipRatio,
			numPressureIters,
			numParticleIters,
			overRelaxation,
			compensateDrift,
			separateParticles,
			damping,
			showGrid
		);
	}

	function render() {
		if (!fluid || !renderer) return;

		renderer.render(fluid, {
			showParticles,
			showFluid,
			showGrid,
			simWidth,
			simHeight
		});
	}

	function update() {
		if (isPageVisible) {
			simulate();
			render();
		}
		animationId = requestAnimationFrame(update);
	}

	onMount(() => {
		const quality = getQualityPreset();
		effectiveResolution = quality.resolution;
		numPressureIters = quality.numPressureIters;
		numParticleIters = quality.numParticleIters;
		damping = quality.damping;
		maxDpr = quality.maxDpr;

		resizeCanvas();

		// Initialize fluid simulation
		fluid = setupFluidScene(
			simWidth,
			simHeight,
			effectiveResolution,
			relWaterWidth,
			relWaterHeight,
			fluidColor,
			foamColor,
			colorDiffusionCoeff,
			foamReturnRate
		);
		renderer = new FluidRenderer(canvas);

		// Initial color is already set via constructor, keep setter for consistency
		if (fluid) {
			fluid.setFluidColor(fluidColor);
			fluid.setFoamColor(foamColor);
			fluid.setColorDiffusionCoeff(colorDiffusionCoeff);
			fluid.setFoamReturnRate(foamReturnRate);
		}

		// Handle window resize
		const handleResize = () => {
			const preset = getQualityPreset();
			maxDpr = preset.maxDpr;
			resizeCanvas();
		};
		const handleVisibilityChange = () => {
			isPageVisible = !document.hidden;
		};
		window.addEventListener('resize', handleResize);
		document.addEventListener('visibilitychange', handleVisibilityChange);

		// Start animation loop
		update();

		return () => {
			window.removeEventListener('resize', handleResize);
			document.removeEventListener('visibilitychange', handleVisibilityChange);
			if (animationId) {
				cancelAnimationFrame(animationId);
			}
		};
	});

	// Watch for color changes and update fluid (supports live changes later)
	$effect(() => {
		if (fluid) {
			fluid.setFluidColor(fluidColor);
		}
	});

	// Watch for foam color changes
	$effect(() => {
		if (fluid) {
			fluid.setFoamColor(foamColor);
		}
	});

	// Watch for diffusion coefficient changes
	$effect(() => {
		if (fluid) {
			fluid.setColorDiffusionCoeff(colorDiffusionCoeff);
		}
	});

	// Watch for foam return rate changes
	$effect(() => {
		if (fluid) {
			fluid.setFoamReturnRate(foamReturnRate);
		}
	});
</script>

<canvas bind:this={canvas} class="absolute inset-0 z-10 h-full w-full"></canvas>
