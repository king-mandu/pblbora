// script.js (ONNX Runtime Web 최종 버전)

// HTML 요소 가져오기
const videoUpload = document.getElementById('videoUpload');
const videoPlayer = document.getElementById('videoPlayer');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true }); // willReadFrequently 추가
const resultsDiv = document.getElementById('results');
const loader = document.getElementById('loader');

const IMAGE_SIZE = 64; // 모델 생성 시 사용했던 이미지 크기 (중요!)

// ONNX Runtime Inference Session (모델) 변수
let session;

// 페이지가 처음 로드될 때 모델을 미리 불러옵니다.
(async () => {
    try {
        resultsDiv.innerText = 'AI 모델을 로드하는 중...';
        // GitHub에 업로드한 ganomaly.onnx 파일 경로
        session = await ort.InferenceSession.create('./ganomaly.onnx');
        resultsDiv.innerText = '모델 로드 완료. 비디오를 업로드하세요.';
    } catch (e) {
        console.error(`모델 로드 실패: ${e}`);
        resultsDiv.innerText = '모델 로드에 실패했습니다. 페이지를 새로고침해주세요.';
    }
})();

// 사용자가 비디오 파일을 선택하면 이 함수가 실행됩니다.
videoUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file || !session) return;

    resultsDiv.innerText = '';
    loader.style.display = 'block';

    // 파일을 URL로 변환하여 video 요소에 로드
    const fileURL = URL.createObjectURL(file);
    videoPlayer.src = fileURL;

    // 비디오가 재생 가능 상태가 되면 프레임 분석 시작
    videoPlayer.onloadeddata = async () => {
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        
        const videoDuration = videoPlayer.duration;
        let maxAnomalyScore = 0;
        const framesPerSecond = 2; // 초당 2개 프레임 분석 (성능에 따라 조절)

        // 비디오 전체를 돌며 프레임 추출 및 분석 (fream화)
        for (let time = 0; time < videoDuration; time += 1 / framesPerSecond) {
            videoPlayer.currentTime = time;
            await new Promise(resolve => { videoPlayer.onseeked = resolve; });

            // 캔버스에 현재 비디오 프레임 그리기 (크기 조절 포함)
            ctx.drawImage(videoPlayer, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
            
            // ONNX Runtime으로 프레임 분석
            const anomalyScore = await analyzeFrameWithONNX();
            if (anomalyScore > maxAnomalyScore) {
                maxAnomalyScore = anomalyScore;
            }
        }
        
        loader.style.display = 'none';
        
        // 최종 결과 표시 (임계값은 실험을 통해 조절)
        const threshold = 0.08; 
        if (maxAnomalyScore > threshold) {
            resultsDiv.innerText = `⚠️ 딥페이크 의심 (최대 비정상 점수: ${maxAnomalyScore.toFixed(4)})`;
            resultsDiv.style.color = 'red';
        } else {
            resultsDiv.innerText = `✅ 정상 영상으로 보입니다 (최대 비정상 점수: ${maxAnomalyScore.toFixed(4)})`;
            resultsDiv.style.color = 'green';
        }
    };
});

// 캔버스 이미지를 텐서로 변환하고 ONNX Runtime으로 분석하는 함수
async function analyzeFrameWithONNX() {
    // 캔버스에서 이미지 데이터(R,G,B,A 값 배열) 가져오기
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    // ONNX Runtime이 요구하는 Float32Array 형태로 데이터 변환 및 정규화
    const float32Data = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);
    for (let i = 0; i < imageData.data.length / 4; i++) {
        const R = imageData.data[i * 4];
        const G = imageData.data[i * 4 + 1];
        const B = imageData.data[i * 4 + 2];
        
        // C, H, W 순서로 데이터 배치 및 -1~1 사이로 정규화
        float32Data[i] = (R / 255.0 - 0.5) * 2;
        float32Data[i + IMAGE_SIZE * IMAGE_SIZE] = (G / 255.0 - 0.5) * 2;
        float32Data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = (B / 255.0 - 0.5) * 2;
    }

    // ONNX Runtime 입력 텐서 생성
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
    
    // 모델 실행 (입력 이름은 모델에 따라 다를 수 있음)
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const reconstructedTensor = results[session.outputNames[0]]; // 복원된 이미지 텐서

    // 비정상 점수 계산 (L1 거리)
    let anomalyScore = 0;
    for (let i = 0; i < float32Data.length; i++) {
        anomalyScore += Math.abs(float32Data[i] - reconstructedTensor.data[i]);
    }
    return anomalyScore / float32Data.length;

}
