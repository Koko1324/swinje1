import streamlit as st
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
if 'is_studying' not in st.session_state:
    st.session_state.is_studying = False
if 'last_face_detected_time' not in st.session_state:
    st.session_state.last_face_detected_time = None
if 'is_sleeping' not in st.session_state:
    st.session_state.is_sleeping = False
if 'db_level' not in st.session_state:
    st.session_state.db_level = 0
if 'high_db_start_time' not in st.session_state:
    st.session_state.high_db_start_time = None
if 'is_warning_shown' not in st.session_state:
    st.session_state.is_warning_shown = False
if 'no_face_detected_start_time' not in st.session_state:
    st.session_state.no_face_detected_start_time = None
if 'accumulated_sleep_time' not in st.session_state:
    st.session_state.accumulated_sleep_time = 0
if 'total_study_time' not in st.session_state:
    st.session_state.total_study_time = 0

def display_timer():
    """ 타이머를 계산하여 총 공부 시간과 잠을 잔 시간을 포맷하여 반환하는 함수 """
    if st.session_state.start_time is None:
        return "00:00:00", "00:00:00"
    elapsed_time = time.time() - st.session_state.start_time
    sleep_time = st.session_state.accumulated_sleep_time
    total_study_time = elapsed_time - sleep_time
    return time.strftime('%H:%M:%S', time.gmtime(total_study_time)), time.strftime('%H:%M:%S', time.gmtime(sleep_time))

# 웹캠과 마이크 권한 요청 함수
def request_permissions():
    """ 웹캠과 마이크 접근 권한을 요청하는 함수 """
    st.write("웹캠과 마이크 접근 권한을 허용해 주세요.")
    cap = cv2.VideoCapture(0)  # 웹캠 접근 시도
    if cap.isOpened():
        st.write("웹캠 접근이 허용되었습니다.")
        cap.release()
    else:
        st.error("웹캠 접근 권한이 거부되었습니다. 권한을 허용해 주세요.")
        return False

    try:
        with sd.InputStream(callback=lambda *args: None):  # 오디오 접근 시도
            st.write("마이크 접근이 허용되었습니다.")
    except Exception as e:
        st.error("마이크 접근 권한이 거부되었습니다. 권한을 허용해 주세요.")
        return False

    return True

# 공부 시작하기 버튼 클릭 시
if st.button("공부 시작하기", key="start_button"):
    
    if request_permissions():  # 권한 요청
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
        st.session_state.is_studying = False  # 공부 중 상태 해제
        total_study_time_str, sleep_time_str = display_timer()
        with st.container():
            st.write("공부를 종료합니다!")
            st.write(f"총 공부 시간: {total_study_time_str}")
            st.write(f"잠을 잔 시간: {sleep_time_str}")

# 웹캠과 오디오 스트림을 동시에 처리하는 함수
def process_camera_and_audio():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return

    def calculate_decibel_level(audio_data):
        if len(audio_data) == 0:
            return 0
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms) if rms > 0 else 0

    def audio_callback(indata, frames, time, status):
        if status:
            st.warning(f"Audio stream status: {status}")
        audio_data = indata[:, 0]
        if len(audio_data) > 0:
            db_level = calculate_decibel_level(audio_data)
            st.session_state.db_level = db_level
            
            if db_level > 80:
                if st.session_state.high_db_start_time is None:
                    st.session_state.high_db_start_time = time.time()
                elif time.time() - st.session_state.high_db_start_time > 120:
                    if not st.session_state.is_warning_shown:
                        st.session_state.is_warning_shown = True
                        st.write("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
            else:
                st.session_state.high_db_start_time = None

    stframe = st.empty()
    timer_placeholder = st.empty()
    db_placeholder = st.empty()

    cap.set(cv2.CAP_PROP_FPS, 24)

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            st.write("Listening and watching...")
            while st.session_state.is_studying:
                ret, frame = cap.read()
                if not ret:
                    st.write("비디오 프레임을 읽을 수 없습니다.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    st.session_state.last_face_detected_time = time.time()
                    st.session_state.is_sleeping = False
                    st.session_state.no_face_detected_start_time = None
                    cv2.putText(frame, "공부 중임", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    if st.session_state.no_face_detected_start_time is None:
                        st.session_state.no_face_detected_start_time = time.time()
                    else:
                        time_since_last_detection = time.time() - st.session_state.no_face_detected_start_time
                        if time_since_last_detection >= 5:
                            st.session_state.accumulated_sleep_time += time_since_last_detection
                            st.session_state.no_face_detected_start_time = time.time()
                        cv2.putText(frame, "공부 중이 아님", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                total_study_time_str, sleep_time_str = display_timer()
                with st.container():
                    timer_placeholder.metric(label="총 공부 시간", value=total_study_time_str)
                    timer_placeholder.metric(label="잠을 잔 시간", value=sleep_time_str)
                    db_placeholder.write(f"데시벨: {st.session_state.db_level:.2f} dB")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sd.stop()

# 웹캠만 처리하는 함수
def process_camera_only():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return

    stframe = st.empty()
    timer_placeholder = st.empty()

    cap.set(cv2.CAP_PROP_FPS, 24)

    try:
        while st.session_state.is_studying:
            ret, frame = cap.read()
            if not ret:
                st.write("비디오 프레임을 읽을 수 없습니다.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                st.session_state.last_face_detected_time = time.time()
                st.session_state.is_sleeping = False
                st.session_state.no_face_detected_start_time = None
                cv2.putText(frame, "공부 중임", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if st.session_state.no_face_detected_start_time is None:
                    st.session_state.no_face_detected_start_time = time.time()
                else:
                    time_since_last_detection = time.time() - st.session_state.no_face_detected_start_time
                    if time_since_last_detection >= 5:
                        st.session_state.accumulated_sleep_time += time_since_last_detection
                        st.session_state.no_face_detected_start_time = time.time()
                    cv2.putText(frame, "공부 중이 아님", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            total_study_time_str, sleep_time_str = display_timer()
            with st.container():
                timer_placeholder.metric(label="총 공부 시간", value=total_study_time_str)
                timer_placeholder.metric(label="잠을 잔 시간", value=sleep_time_str)

    finally:
        cap.release()
        cv2.destroyAllWindows()

# 오디오만 처리하는 함수
def process_audio_only():
    def calculate_decibel_level(audio_data):
        if len(audio_data) == 0:
            return 0
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms) if rms > 0 else 0

    def audio_callback(indata, frames, time, status):
        if status:
            st.warning(f"Audio stream status: {status}")
        audio_data = indata[:, 0]
        if len(audio_data) > 0:
            db_level = calculate_decibel_level(audio_data)
            st.session_state.db_level = db_level
            
            if db_level > 80:
                if st.session_state.high_db_start_time is None:
                    st.session_state.high_db_start_time = time.time()
                elif time.time() - st.session_state.high_db_start_time > 120:
                    if not st.session_state.is_warning_shown:
                        st.session_state.is_warning_shown = True
                        st.write("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
            else:
                st.session_state.high_db_start_time = None

    db_placeholder = st.empty()

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            st.write("Listening...")
            while st.session_state.is_studying:
                db_placeholder.write(f"데시벨: {st.session_state.db_level:.2f} dB")

    finally:
        sd.stop()


# 선택된 옵션에 따라 적절한 함수를 호출하여 처리
if st.session_state.is_studying:
    if option == "캠 공부":
        process_camera_only()
    elif option == "데시벨 공부":
        process_audio_only()
    elif option == "캠+데시벨 공부":
        process_camera_and_audio()
