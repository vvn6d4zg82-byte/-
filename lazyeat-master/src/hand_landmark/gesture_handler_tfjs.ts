import { HandGesture, HandInfo } from '@/hand_landmark/detector_tfjs';
import i18n from '@/locales/i18n';
import use_app_store from '@/store/app';

enum WsDataType {
  INFO = 'info',
  SUCCESS = 'success',
  WARNING = 'warning',
  ERROR = 'error',
  MOUSE_MOVE = 'mouse_move',
  MOUSE_CLICK = 'mouse_click',
  MOUSE_SCROLL_UP = 'mouse_scroll_up',
  MOUSE_SCROLL_DOWN = 'mouse_scroll_down',
  SEND_KEYS = 'send_keys',
  VOICE_RECORD = 'voice_record',
  VOICE_STOP = 'voice_stop',
}

interface WsData {
  type: WsDataType;
  msg?: string;
  duration?: number;
  title?: string;
  data?: {
    x?: number;
    y?: number;
    key_str?: string;
  };
}

export class TriggerAction {
  private ws: WebSocket | null = null;

  constructor() {
    this.connectWebSocket();
  }

  private connectWebSocket() {
    try {
      this.ws = new WebSocket('ws://127.0.0.1:62334/ws_lazyeat');
      this.ws.onmessage = (event: MessageEvent) => {
        const response: WsData = JSON.parse(event.data);
        const app_store = use_app_store();
        app_store.sub_window_info(response.msg || '');
      };
      this.ws.onopen = () => {
        console.log('ws_lazyeat connected');
      };
      this.ws.onclose = () => {
        console.log('ws_lazyeat closed, retrying...');
        this.ws = null;
        setTimeout(() => this.connectWebSocket(), 3000);
      };
      this.ws.onerror = (error) => {
        console.error('ws_lazyeat error:', error);
        this.ws?.close();
      };
    } catch (error) {
      console.error('Failed to create WebSocket instance:', error);
      this.ws = null;
      setTimeout(() => this.connectWebSocket(), 1000);
    }
  }

  private send(data: { type: WsDataType } & Partial<Omit<WsData, 'type'>>) {
    const message: WsData = {
      type: data.type,
      msg: data.msg || '',
      title: data.title || 'Lazyeat',
      duration: data.duration || 1,
      data: data.data || {},
    };
    this.ws?.send(JSON.stringify(message));
  }

  moveMouse(x: number, y: number) {
    this.send({
      type: WsDataType.MOUSE_MOVE,
      data: { x, y },
    });
  }

  clickMouse() {
    this.send({
      type: WsDataType.MOUSE_CLICK,
    });
  }

  scrollUp() {
    this.send({
      type: WsDataType.MOUSE_SCROLL_UP,
    });
  }

  scrollDown() {
    this.send({
      type: WsDataType.MOUSE_SCROLL_DOWN,
    });
  }

  sendKeys(key_str: string) {
    this.send({
      type: WsDataType.SEND_KEYS,
      data: { key_str },
    });
  }

  voiceRecord() {
    this.send({
      type: WsDataType.VOICE_RECORD,
    });
  }

  voiceStop() {
    this.send({
      type: WsDataType.VOICE_STOP,
    });
  }
}

export class GestureHandler {
  private triggerAction: TriggerAction;
  private previousGesture: HandGesture | null = null;
  private previousGestureCount: number = 0;
  private minGestureCount: number = 5;

  private screen_width: number = window.screen.width;
  private screen_height: number = window.screen.height;
  private smoothening = 7;
  private prev_loc_x: number = 0;
  private prev_loc_y: number = 0;
  private prev_three_fingers_y: number = 0;
  private prev_scroll2_y: number = 0;

  private lastClickTime: number = 0;
  private lastScrollTime: number = 0;
  private lastFullScreenTime: number = 0;
  private lastDeleteTime: number = 0;

  private readonly CLICK_INTERVAL = 500;
  private readonly SCROLL_INTERVAL = 100;
  private readonly FULL_SCREEN_INTERVAL = 1500;

  private voice_recording: boolean = false;
  private app_store: any;

  constructor() {
    this.triggerAction = new TriggerAction();
    this.app_store = use_app_store();
  }

  private getFingerTip(hand: HandInfo, fingerIndex: number) {
    if (!hand) return null;
    const tipIndices = [4, 8, 12, 16, 20];
    return hand.landmarks[tipIndices[fingerIndex]];
  }

  private mapRange(value: number, fromMin: number, fromMax: number, toMin: number, toMax: number): number {
    return ((value - fromMin) * (toMax - toMin)) / (fromMax - fromMin) + toMin;
  }

  private handleIndexFingerUp(hand: HandInfo) {
    const indexTip = this.getFingerTip(hand, 1);
    if (!indexTip) return;

    try {
      const video_x = indexTip.x * this.app_store.VIDEO_WIDTH;
      const video_y = indexTip.y * this.app_store.VIDEO_HEIGHT;

      let screenX = this.mapRange(
        video_x,
        this.app_store.config.boundary_left,
        this.app_store.config.boundary_left + this.app_store.config.boundary_width,
        0,
        this.screen_width
      );

      let screenY = this.mapRange(
        video_y,
        this.app_store.config.boundary_top,
        this.app_store.config.boundary_top + this.app_store.config.boundary_height,
        0,
        this.screen_height
      );

      screenX = this.prev_loc_x + (screenX - this.prev_loc_x) / this.smoothening;
      screenY = this.prev_loc_y + (screenY - this.prev_loc_y) / this.smoothening;

      this.prev_loc_x = screenX;
      this.prev_loc_y = screenY;

      this.app_store.sub_windows.x = screenX + 10;
      this.app_store.sub_windows.y = screenY;
      this.triggerAction.moveMouse(screenX, screenY);
    } catch (error) {
      console.error('处理鼠标移动失败:', error);
    }
  }

  private handleMouseClick() {
    const now = Date.now();
    if (now - this.lastClickTime < this.CLICK_INTERVAL) {
      return;
    }
    this.lastClickTime = now;
    this.triggerAction.clickMouse();
  }

  private handleScroll2(hand: HandInfo) {
    const indexTip = this.getFingerTip(hand, 1);
    const thumbTip = this.getFingerTip(hand, 0);
    if (!indexTip || !thumbTip) {
      this.prev_scroll2_y = 0;
      return;
    }

    const now = Date.now();
    if (now - this.lastScrollTime < this.SCROLL_INTERVAL) {
      return;
    }
    this.lastScrollTime = now;

    const distance = Math.sqrt(
      (indexTip.x - thumbTip.x) ** 2 + (indexTip.y - thumbTip.y) ** 2
    );

    if (distance > this.app_store.config.scroll_gesture_2_thumb_and_index_threshold) {
      this.prev_scroll2_y = 0;
      return;
    }

    if (this.prev_scroll2_y === 0) {
      this.prev_scroll2_y = indexTip.y;
      return;
    }

    const deltaY = indexTip.y - this.prev_scroll2_y;

    if (Math.abs(deltaY) > 0.008) {
      if (deltaY < 0) {
        this.triggerAction.scrollUp();
      } else {
        this.triggerAction.scrollDown();
      }
      this.prev_scroll2_y = indexTip.y;
    }
  }

  private handleFourFingers() {
    try {
      const key_str = this.app_store.config.four_fingers_up_send || 'f';
      const now = Date.now();
      if (now - this.lastFullScreenTime < this.FULL_SCREEN_INTERVAL) {
        return;
      }
      this.lastFullScreenTime = now;
      this.triggerAction.sendKeys(key_str);
    } catch (error) {
      console.error('处理四指手势失败:', error);
    }
  }

  async handleVoiceStart() {
    if (this.voice_recording) {
      return;
    }
    await this.app_store.sub_window_info('开始语音识别');
    this.voice_recording = true;
    this.triggerAction.voiceRecord();
  }

  async handleVoiceStop() {
    if (!this.voice_recording) {
      return;
    }
    await this.app_store.sub_window_success('停止语音识别');
    this.voice_recording = false;
    this.triggerAction.voiceStop();
  }

  private handleDelete() {
    const now = Date.now();
    if (now - this.lastDeleteTime < 300) {
      return;
    }
    this.lastDeleteTime = now;
    this.triggerAction.sendKeys('backspace');
  }

  async handleStopGesture(): Promise<void> {
    const toogle_detect = () => {
      this.app_store.flag_detecting = !this.app_store.flag_detecting;
    };

    if (this.previousGestureCount >= 60) {
      toogle_detect();
      this.previousGestureCount = 0;
      this.app_store.sub_windows.progress = 0;
    } else {
      this.previousGestureCount++;
      if (this.previousGestureCount >= 20) {
        this.app_store.sub_windows.progress = Math.floor(
          (this.previousGestureCount / 60) * 100
        );
      }
    }
  }

  handleGesture(gesture: HandGesture, hand: HandInfo) {
    if (gesture === this.previousGesture) {
      this.previousGestureCount++;
    } else {
      this.previousGesture = gesture;
      this.previousGestureCount = 1;
    }

    if (gesture === HandGesture.STOP_GESTURE) {
      if (hand.categoryName === 'Open_Palm') {
        this.handleStopGesture();
      }
      return;
    }

    if (!this.app_store.flag_detecting) {
      return;
    }

    if (gesture !== HandGesture.VOICE_GESTURE_START && this.voice_recording) {
      this.handleVoiceStop();
      return;
    }

    if (gesture === HandGesture.ONLY_INDEX_UP) {
      this.handleIndexFingerUp(hand);
      return;
    }

    if (this.previousGestureCount >= this.minGestureCount) {
      switch (gesture) {
        case HandGesture.ROCK_GESTURE:
        case HandGesture.INDEX_AND_MIDDLE_UP:
          this.handleMouseClick();
          break;
        case HandGesture.SCROLL_GESTURE_2:
          this.handleScroll2(hand);
          break;
        case HandGesture.FOUR_FINGERS_UP:
          this.handleFourFingers();
          break;
        case HandGesture.VOICE_GESTURE_START:
          this.handleVoiceStart();
          break;
        case HandGesture.DELETE_GESTURE:
          this.handleDelete();
          break;
      }
    }
  }
}
