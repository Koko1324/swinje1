import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sounddevice as sd
import time

# Streamlit 페이지 설정
st.set_page_config(layout="wide")

# 로고 이미지와 공부 모드 선택 UI 설정
logo = Image.open("logo.png")
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    st.image(logo, use_column_width=True)
with header_col2:
    option = st.selectbox(
        "공부 모드 선택",
        ["캠 공부", "데시벨 공부", "캠+데시벨 공부"],
        key='selectbox_option'
    )

# 상태 초기화 (타이머와 상태 관리를 위한 세션 상태 변수 설정)
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
#if 'elapsed_time' not in st.session_state:
#   st.session_state.elapsed_time = 0 #경과 시간
if 'is_studying' not in st.session_state:
    st.session_state.is_studying = False
if 'sleep_time' not in st.session_state:
    st.session_state.sleep_time = 0
if 'last_face_detected_time' not in st.session_state:
    st.session_state.last_face_detected_time = None
if 'is_sleeping' not in st.session_state:
    st.session_state.is_sleeping = False
if 'db_level' not in st.session_state:
    st.session_state.db_level = 0
if 'high_db_start_time' not in st.session_state:
    st.session_state.high_db_start_time = None  # 80dB 이상이 시작된 시간을 기록
if 'is_warning_shown' not in st.session_state:
    st.session_state.is_warning_shown = False  # 경고 메시지 출력 여부
if 'no_face_detected_start_time' not in st.session_state:
    st.session_state.no_face_detected_start_time = None  # 얼굴 인식 실패 시작 시간
if 'accumulated_sleep_time' not in st.session_state:
    st.session_state.accumulated_sleep_time = 0  # 누적 잠을 잔 시간
total_study_time = 0

# 모델 로드 (수면 감지 모델)
#model = tf.keras.models.load_model('sleep_detection_model.h5')

def display_timer():
    """ 타이머를 계산하여 총 공부 시간과 잠을 잔 시간을 포맷하여 반환하는 함수 """
    #elapsed_time = st.session_state.elapsed_time + (time.time() - st.session_state.start_time)
    elapsed_time = time.time() - st.session_state.start_time
    sleep_time = st.session_state.accumulated_sleep_time  # 누적 잠을 잔 시간
    total_study_time = elapsed_time - sleep_time
    return time.strftime('%H:%M:%S', time.gmtime(total_study_time)), time.strftime('%H:%M:%S', time.gmtime(sleep_time))

# 공부 시작하기 버튼 클릭 시
if st.button("공부 시작하기", key="start_button"):
    st.session_state.start_time = time.time()  # 시작 시간 기록
    st.session_state.is_studying = True  # 공부 중 상태 설정
    st.session_state.last_face_detected_time = time.time()  # 얼굴 감지 시간 기록
    st.session_state.is_warning_shown = False  # 경고 메시지 초기화
    st.session_state.no_face_detected_start_time = None  # 얼굴 인식 실패 시작 시간 초기화
    st.session_state.accumulated_sleep_time = 0  # 누적 잠을 잔 시간 초기화
    st.write("공부를 시작합니다!")

# 공부 그만하기 버튼 클릭 시
if st.button("공부 그만하기", key="stop_button"):
    if st.session_state.is_studying:
        #st.session_state.elapsed_time += time.time() - st.session_state.start_time  # 총 경과 시간 업데이트
        st.session_state.is_studying = False  # 공부 중 상태 해제
        #total_study_time = st.session_state.elapsed_time - st.session_state.accumulated_sleep_time
        if total_study_time < 0:
            total_study_time = 0
        with st.container(border = True):
            with st.container(border = True):
                st.write("공부를 종료합니다!")
            with st.container(border = True):
                st.write(f"총 공부 시간: {time.strftime('%H:%M:%S', time.gmtime(total_study_time))}")
                st.write(f"잠을 잔 시간: {time.strftime('%H:%M:%S', time.gmtime(st.session_state.accumulated_sleep_time))}")

# 웹캠과 오디오 스트림을 동시에 처리하는 함수
def process_camera_and_audio():
    # 웹캠 얼굴 인식을 위한 설정
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haarcascade불러오기
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return

    # 오디오 설정
    def calculate_decibel_level(audio_data):
        """ 오디오 데이터에서 데시벨 수준을 계산하는 함수 """
        if len(audio_data) == 0:
            return 0
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms) if rms > 0 else 0

    def audio_callback(indata, frames, time, status):
        """ 오디오 스트림에서 호출되는 콜백 함수 """
        if status:
            st.warning(f"Audio stream status: {status}")
        audio_data = indata[:, 0]
        if len(audio_data) > 0:
            db_level = calculate_decibel_level(audio_data)
            st.session_state.db_level = db_level  # 세션 상태에 데시벨 레벨 저장
            
            # 80dB 이상 지속 시간 체크
            if db_level > 80:
                if st.session_state.high_db_start_time is None:
                    st.session_state.high_db_start_time = time.time()  # 시작 시간 기록
                elif time.time() - st.session_state.high_db_start_time > 120:  # 2분 이상
                    if not st.session_state.is_warning_shown:
                        st.session_state.is_warning_shown = True
                        st.write("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
            else:
                st.session_state.high_db_start_time = None  # 80dB 이하로 돌아오면 시작 시간 초기화

    # Streamlit 공간 설정
    stframe = st.empty()  # 웹캠 프레임을 표시할 공간
    timer_placeholder = st.empty()  # 타이머를 표시할 공간
    db_placeholder = st.empty()  # 데시벨 레벨을 표시할 공간

    cap.set(cv2.CAP_PROP_FPS, 24)  # FPS 설정

    #예외 처리
    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            st.write("Listening and watching...")
            while st.session_state.is_studying:
                ret, frame = cap.read()  # 웹캠에서 프레임 읽기
                if not ret:
                    st.write("비디오 프레임을 읽을 수 없습니다.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 얼굴 감지를 위한 회색조 변환
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    st.session_state.last_face_detected_time = time.time()  # 얼굴 감지 시간 갱신
                    st.session_state.is_sleeping = False  # 깨어 있는 상태로 설정
                    st.session_state.no_face_detected_start_time = None  # 얼굴 인식 실패 시작 시간 초기화
                    cv2.putText(frame, "Awake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    if st.session_state.no_face_detected_start_time is None:
                        st.session_state.no_face_detected_start_time = time.time()  # 얼굴 인식 실패 시작 시간 기록
                    else:
                        # 누적 잠을 잔 시간 업데이트
                        time_since_last_detection = time.time() - st.session_state.no_face_detected_start_time
                        if time_since_last_detection >= 5:  # 5초 이상 얼굴이 감지되지 않으면
                            st.session_state.accumulated_sleep_time += time_since_last_detection
                            st.session_state.no_face_detected_start_time = time.time()  # 새로운 측정을 위해 시작 시간 갱신
                        cv2.putText(frame, "Sleeping or Not Present", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)  # 웹캠 프레임을 Streamlit에 표시
                total_study_time_str, sleep_time_str = display_timer()  # 타이머 업데이트
                timer_placeholder.metric(label="총 공부 시간", value=total_study_time_str)
                timer_placeholder.metric(label="잠을 잔 시간", value=sleep_time_str)
                db_placeholder.write(f"Decibel Level: {st.session_state.db_level:.2f} dB")  # 데시벨 레벨 업데이트

    finally:
        cap.release()  # 웹캠 자원 해제
        cv2.destroyAllWindows()  # OpenCV 창 닫기
        sd.stop()  # 오디오 스트림 중지

# 선택된 모드에 따라 기능 실행
if option == "캠 공부" or option == "캠+데시벨 공부":
    if st.session_state.is_studying:
        process_camera_and_audio()  # 웹캠과 오디오 스트림 처리

elif option == "데시벨 공부":
    if st.session_state.is_studying:
        def calculate_decibel_level(audio_data):
            """ 오디오 데이터에서 데시벨 수준을 계산하는 함수 """
            if len(audio_data) == 0:
                return 0
            rms = np.sqrt(np.mean(np.square(audio_data)))
            return 20 * np.log10(rms) if rms > 0 else 0

        def audio_callback(indata, frames, time, status):
            """ 오디오 스트림에서 호출되는 콜백 함수 """
            if status:
                st.warning(f"Audio stream status: {status}")
            audio_data = indata[:, 0]
            if len(audio_data) > 0:
                db_level = calculate_decibel_level(audio_data)
                st.session_state.db_level = db_level
                if db_level > 1:
                    if st.session_state.high_db_start_time is None:
                        st.session_state.high_db_start_time = time.time()
                    elif time.time() - st.session_state.high_db_start_time > 120:
                        if not st.session_state.is_warning_shown:
                            st.session_state.is_warning_shown = True
                            st.write("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
                else:
                    st.session_state.high_db_start_time = None

        db_placeholder = st.empty()  # 데시벨 레벨을 표시할 공간
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
                st.write("Listening for decibels...")
                while st.session_state.is_studying:
                    db_placeholder.write(f"Decibel Level: {st.session_state.db_level:.2f} dB")  # 데시벨 레벨 업데이트
                    time.sleep(0.5)  # 0.5초마다 데시벨 레벨 업데이트

        except Exception as e:
            st.error(f"An error occurred: {e}")  # 예외 발생 시 오류 메시지 출력

        finally:
            sd.stop()  # 오디오 스트림 중지
