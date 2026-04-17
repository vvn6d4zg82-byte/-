import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';

export enum HandGesture {
  ONLY_INDEX_UP = 'only_index_up',
  INDEX_AND_MIDDLE_UP = 'index_and_middle_up',
  ROCK_GESTURE = 'rock_gesture',
  THREE_FINGERS_UP = 'three_fingers_up',
  SCROLL_GESTURE_2 = 'scroll_gesture_2',
  FOUR_FINGERS_UP = 'four_fingers_up',
  STOP_GESTURE = 'stop_gesture',
  VOICE_GESTURE_START = 'voice_gesture_start',
  VOICE_GESTURE_STOP = 'voice_gesture_stop',
  DELETE_GESTURE = 'delete_gesture',
  OTHER = 'other',
}

export interface HandLandmark {
  x: number;
  y: number;
  z?: number;
  name?: string;
}

export interface HandInfo {
  landmarks: HandLandmark[];
  handedness: 'Left' | 'Right';
  score: number;
  categoryName?: string;
  keypoints?: handPoseDetection.Keypoint[];
  keypoints3D?: handPoseDetection.Keypoint3D[];
}

interface DetectionResult {
  leftHand?: HandInfo;
  rightHand?: HandInfo;
  rawResult: handPoseDetection.Hand[];
}

export class Detector {
  private detector: handPoseDetection.HandPoseDetector | null = null;
  private gestureHandler: any = null;

  async initialize(useCanvas = false) {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig: handPoseDetection.MediaPipeHandsMediaPipeModelConfig = {
      runtime: 'tfjs',
      modelType: 'full',
      maxHands: 1,
      detectorModelUrl: undefined,
      landmarkModelUrl: undefined,
    };

    this.detector = await handPoseDetection.createDetector(model, detectorConfig);
    
    const { GestureHandler } = await import('@/hand_landmark/gesture_handler_tfjs');
    this.gestureHandler = new GestureHandler();
  }

  async detect(video: HTMLVideoElement): Promise<DetectionResult> {
    if (!this.detector) {
      throw new Error('检测器未初始化');
    }

    const hands = await this.detector.estimateHands(video);
    const detection: DetectionResult = {
      rawResult: hands,
    };

    if (hands.length > 0) {
      for (const hand of hands) {
        const handInfo: HandInfo = {
          landmarks: hand.keypoints.map(kp => ({
            x: kp.x / video.width,
            y: kp.y / video.height,
            z: kp.z || 0,
            name: kp.name,
          })),
          handedness: hand.handedness as 'Left' | 'Right',
          score: hand.score || 0,
          keypoints: hand.keypoints,
          keypoints3D: hand.keypoints3D,
        };

        if (hand.handedness === 'Left') {
          detection.leftHand = handInfo;
        } else {
          detection.rightHand = handInfo;
        }
      }
    }

    return detection;
  }

  static getFingerLandmarks(hand: HandInfo | undefined, fingerIndex: number): HandLandmark[] | null {
    if (!hand) return null;

    const fingerIndices = {
      thumb: [1, 2, 3, 4],
      index: [5, 6, 7, 8],
      middle: [9, 10, 11, 12],
      ring: [13, 14, 15, 16],
      pinky: [17, 18, 19, 20],
    };

    const indices = Object.values(fingerIndices)[fingerIndex];
    return indices.map((i) => hand.landmarks[i]);
  }

  static getFingerTip(hand: HandInfo | undefined, fingerIndex: number): HandLandmark | null {
    if (!hand) return null;

    const tipIndices = [4, 8, 12, 16, 20];
    return hand.landmarks[tipIndices[fingerIndex]];
  }

  static _fingersUp(hand: HandInfo): number[] {
    const fingers: number[] = [];
    const tipIds = [4, 8, 12, 16, 20];
    const thumbBaseId = 2;
    const indexPipId = 6;
    const middlePipId = 10;
    const ringPipId = 14;
    const pinkyPipId = 18;

    if (hand.handedness === 'Right') {
      if (hand.landmarks[tipIds[0]].x < hand.landmarks[thumbBaseId].x) {
        fingers.push(0);
      } else {
        fingers.push(1);
      }
    } else {
      if (hand.landmarks[tipIds[0]].x > hand.landmarks[thumbBaseId].x) {
        fingers.push(0);
      } else {
        fingers.push(1);
      }
    }

    const pipIds = [indexPipId, middlePipId, ringPipId, pinkyPipId];
    for (let i = 0; i < 4; i++) {
      if (hand.landmarks[tipIds[i + 1]].y < hand.landmarks[pipIds[i]].y) {
        fingers.push(1);
      } else {
        fingers.push(0);
      }
    }

    return fingers;
  }

  public static getSingleHandGesture(hand: HandInfo): HandGesture {
    const fingers = this._fingersUp(hand);
    const fingerState = fingers.join(',');

    const gestureMap = new Map<string, HandGesture>([
      ['0,1,0,0,0', HandGesture.ONLY_INDEX_UP],
      ['0,1,1,0,0', HandGesture.INDEX_AND_MIDDLE_UP],
      ['0,1,0,0,1', HandGesture.ROCK_GESTURE],
      ['1,1,0,0,1', HandGesture.ROCK_GESTURE],
      ['0,1,1,1,0', HandGesture.THREE_FINGERS_UP],
      ['1,0,1,1,1', HandGesture.SCROLL_GESTURE_2],
      ['0,0,1,1,1', HandGesture.SCROLL_GESTURE_2],
      ['0,1,1,1,1', HandGesture.FOUR_FINGERS_UP],
      ['1,1,1,1,1', HandGesture.STOP_GESTURE],
      ['1,0,0,0,1', HandGesture.VOICE_GESTURE_START],
      ['0,0,0,0,0', HandGesture.VOICE_GESTURE_STOP],
    ]);

    if (gestureMap.has(fingerState)) {
      return gestureMap.get(fingerState) as HandGesture;
    }

    if (this._isDeleteGesture(hand, fingers)) {
      return HandGesture.DELETE_GESTURE;
    }

    return HandGesture.OTHER;
  }

  private static _isDeleteGesture(hand: HandInfo, fingers: number[]): boolean {
    const THUMB_INDEX = 4;
    const FINGER_TIPS = [8, 12, 16, 20];
    const distance_threshold = 0.05;

    const isThumbExtended = fingers[0] === 1;
    const areOtherFingersClosed = fingers.slice(1).every((finger) => finger === 0);
    const isThumbLeftmost = FINGER_TIPS.every(
      (tipIndex) => hand.landmarks[THUMB_INDEX].x > hand.landmarks[tipIndex].x + distance_threshold
    );

    return isThumbExtended && areOtherFingersClosed && isThumbLeftmost;
  }

  async process(detection: DetectionResult): Promise<void> {
    const rightHandGesture = detection.rightHand
      ? Detector.getSingleHandGesture(detection.rightHand)
      : HandGesture.OTHER;
    const leftHandGesture = detection.leftHand
      ? Detector.getSingleHandGesture(detection.leftHand)
      : HandGesture.OTHER;

    let effectiveGesture = rightHandGesture;
    if (detection.rightHand) {
      effectiveGesture = rightHandGesture;
    } else if (detection.leftHand) {
      effectiveGesture = leftHandGesture;
    }

    if (detection.rightHand) {
      this.gestureHandler?.handleGesture(effectiveGesture, detection.rightHand);
    } else if (detection.leftHand) {
      this.gestureHandler?.handleGesture(effectiveGesture, detection.leftHand);
    }
  }
}
