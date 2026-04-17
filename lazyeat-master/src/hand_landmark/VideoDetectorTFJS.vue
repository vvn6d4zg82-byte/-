<template>
  <div v-if="!camera_premission">
    <n-alert title="获取摄像头权限失败" type="error">
      <p>请尝试以下步骤解决:</p>
      <ol>
        <li>检查摄像头是否被其他程序占用</li>
        <li>确保浏览器有摄像头权限</li>
        <li>重启程序</li>
      </ol>
    </n-alert>
  </div>

  <div v-else>
    <span>FPS: {{ FPS }}</span>
    <div class="hand-detection">
      <video
        ref="videoElement"
        class="input-video"
        :width="app_store.VIDEO_WIDTH"
        :height="app_store.VIDEO_HEIGHT"
        autoplay
        style="display: none"
      ></video>
      <canvas
        ref="canvasElement"
        class="output-canvas"
        :width="app_store.VIDEO_WIDTH"
        :height="app_store.VIDEO_HEIGHT"
      ></canvas>
    </div>

    <n-popover trigger="hover" placement="bottom">
      <template #trigger>
        <n-button text circle type="info">
          <n-icon size="24">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"
                fill="currentColor"
              />
            </svg>
          </n-icon>
        </n-button>
      </template>

      <div class="tooltip-content">
        <p><strong>识别框的位置决定了光标的移动范围</strong></p>
        <ul>
          <li>识别框x和y：决定识别框的左上角位置</li>
          <li>当手势在识别框内移动时，光标会跟随手势在屏幕上移动</li>
        </ul>
      </div>
    </n-popover>

    <n-space>
      <n-form-item label="识别框x">
        <n-input-number
          v-model:value="app_store.config.boundary_left"
          :min="0"
          :max="app_store.VIDEO_WIDTH - app_store.config.boundary_width - 10"
          style="width: 150px"
        />
      </n-form-item>

      <n-form-item label="识别框y">
        <n-input-number
          v-model:value="app_store.config.boundary_top"
          :min="0"
          :max="app_store.VIDEO_HEIGHT - app_store.config.boundary_height - 10"
          style="width: 150px"
        />
      </n-form-item>

      <n-form-item label="识别框宽">
        <n-input-number
          v-model:value="app_store.config.boundary_width"
          :min="0"
          :max="app_store.VIDEO_WIDTH - app_store.config.boundary_left - 10"
          style="width: 150px"
        />
      </n-form-item>

      <n-form-item label="识别框高">
        <n-input-number
          v-model:value="app_store.config.boundary_height"
          :min="0"
          :max="app_store.VIDEO_HEIGHT - app_store.config.boundary_top - 10"
          style="width: 150px"
        />
      </n-form-item>
    </n-space>
  </div>
</template>

<script setup lang="ts">
import { Detector } from '@/hand_landmark/detector_tfjs';
import { use_app_store } from '@/store/app';
import { NButton, NIcon, NPopover } from 'naive-ui';
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';

const app_store = use_app_store();

const videoElement = ref<HTMLVideoElement | null>(null);
const canvasElement = ref<HTMLCanvasElement | null>(null);
const detector = ref<Detector | null>(null);
const lastVideoTime = ref(-1);
const currentStream = ref<MediaStream | null>(null);
const FPS = ref(0);
const camera_premission = ref(false);

onMounted(() => {
  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    camera_premission.value = true;
    stream.getTracks().forEach((track) => track.stop());
  });
});

const drawMouseMoveBox = (ctx: CanvasRenderingContext2D) => {
  ctx.strokeStyle = 'rgb(255, 0, 255)';
  ctx.lineWidth = 2;
  ctx.strokeRect(
    app_store.config.boundary_left,
    app_store.config.boundary_top,
    app_store.config.boundary_width,
    app_store.config.boundary_height
  );
};

const frameCount = ref(0);
const fpsUpdateInterval = 1000;
const lastFpsTime = ref(0);

const drawFPS = (ctx: CanvasRenderingContext2D) => {
  const now = performance.now();
  frameCount.value++;

  if (now - lastFpsTime.value >= fpsUpdateInterval) {
    FPS.value = frameCount.value;
    frameCount.value = 0;
    lastFpsTime.value = now;
  }
};

const drawHandLandmarks = (ctx: CanvasRenderingContext2D, hand: any, color: string) => {
  hand.keypoints?.forEach((landmark: any) => {
    ctx.beginPath();
    ctx.arc(landmark.x, landmark.y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  });
};

const predictWebcam = async () => {
  const video = videoElement.value;
  const canvas = canvasElement.value;
  if (!video || !canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  if (video.currentTime !== lastVideoTime.value) {
    lastVideoTime.value = video.currentTime;
    
    if (detector.value) {
      const detection = await detector.value.detect(video);

      if (app_store.config.show_window) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (detection.leftHand) {
          drawHandLandmarks(ctx, detection.leftHand, 'red');
        }
        if (detection.rightHand) {
          drawHandLandmarks(ctx, detection.rightHand, 'blue');
        }

        ctx.restore();
        drawMouseMoveBox(ctx);
      }

      await detector.value.process(detection);
      drawFPS(ctx);
    }
  }

  requestAnimationFrame(predictWebcam);
};

const initializeCamera = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: app_store.config.selected_camera_id
          ? { exact: app_store.config.selected_camera_id }
          : undefined,
        width: { ideal: app_store.VIDEO_WIDTH },
        height: { ideal: app_store.VIDEO_HEIGHT },
      },
      audio: false,
    });
    currentStream.value = stream;
    if (videoElement.value) {
      videoElement.value.srcObject = stream;
      videoElement.value.addEventListener('loadeddata', predictWebcam);
    }
  } catch (error) {
    console.error('无法访问摄像头:', error);
  }
};

const stopCamera = () => {
  if (videoElement.value?.srcObject) {
    (videoElement.value.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
  }
};

watch(
  () => app_store.config.selected_camera_id,
  async () => {
    stopCamera();
  }
);

watch(
  () => app_store.mission_running,
  async (newValue) => {
    if (newValue) {
      await initializeCamera();
      app_store.flag_detecting = true;
    } else {
      stopCamera();
    }
  }
);

onMounted(async () => {
  detector.value = new Detector();
  await detector.value.initialize();
  if (app_store.mission_running) {
    await initializeCamera();
  }
});

onBeforeUnmount(() => {
  stopCamera();
});
</script>

<style scoped>
.hand-detection {
  width: v-bind('app_store.VIDEO_WIDTH + "px"');
  height: v-bind('app_store.VIDEO_HEIGHT + "px"');
}

.output-canvas {
  position: absolute;
}

.tooltip-content {
  max-width: 300px;
  padding: 4px;
}

.tooltip-content p {
  margin-top: 0;
}

.tooltip-content ul {
  margin-bottom: 0;
  padding-left: 20px;
}
</style>
