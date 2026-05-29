import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

from feat_space_analysis.lib.eval_plots import _draw_850hpa_on_ax

def navigate_on_feats_temporal(Z_all, file_list, start_i=0):
    # =========================
    # 고정 grid
    # =========================
    lon = np.arange(105.25, 147, 0.25)   # 180
    lat = np.arange(58.75, 16, -0.25)    # 180
    lon2d, lat2d = np.meshgrid(lon, lat)

    def load_raw(i):
        one_file = file_list[i]
        one_img = np.load(one_file)

        t = one_img[:, :, 0]
        z = one_img[:, :, 1]
        u = one_img[:, :, 2]
        v = one_img[:, :, 3]

        return t, z, u, v
    
    temp_levels = np.arange(-30, 31, 3)   # 고정
    state = {
        "drag": False,
        "idx": None,
        "cbar": None,
        "cax": None, #fig.add_axes([0.90, 0.15, 0.02, 0.7]),
    }

    # =========================
    # figure
    # =========================
    fig = plt.figure(2, figsize=(12, 6))
    ax_feat = fig.add_subplot(121)
    ax_map = fig.add_subplot(122)
    
    # fig.subplots_adjust(right=0.88)
    # ax_feat = fig.add_subplot(121)
    # ax_map = fig.add_subplot(122, projection=ccrs.PlateCarree())

    state["cax"] = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    sc = ax_feat.scatter(
        Z_all[:, 0],
        Z_all[:, 1],
        s=4,
        c="gray",
        alpha=0.5
    )

    selected_sc = ax_feat.scatter(
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2
    )

    ax_feat.set_title("Feature space")
    ax_map.set_title("850 hPa map")

    def idx_to_datestr(i, start=datetime(2020, 1, 1, 0, 0), step_hours=6):
        dt = start + timedelta(hours=int(i) * int(step_hours))
        return dt.strftime("%Y-%m-%d %H:%M")
    
    def draw_map(i):
        t, z, u, v = load_raw(i)
        ax_map.cla()
        #draw_base_map(ax_map, lon, lat)    

        cf_last = _draw_850hpa_on_ax(
            ax_map,
            lon2d,
            lat2d,
            t,
            z,
            u,
            v,
            use_theta=False,
            barb_skip=8,
            title=f"idx = {idx_to_datestr(i)}",
            shaded_levels=temp_levels,
        )
        # colorbar는 처음 한 번만 생성
        if (cf_last is not None) and (state["cbar"] is None):
            state["cbar"] = fig.colorbar(cf_last, cax=state["cax"])
            state["cbar"].set_label("Temperature (°C)")

    N = len(file_list)
    for i in range(start_i, N):
        state["idx"] = i
        # 표시 점 업데이트
        selected_sc.set_offsets(Z_all[i])
        # 지도 업데이트
        draw_map(i)
        plt.pause(0.5)

def navigate_on_feats(Z_all, file_list):    
    # =========================
    # 고정 grid
    # =========================
    lon = np.arange(105.25, 147, 0.25)   # 180
    lat = np.arange(58.75, 16, -0.25)    # 180
    lon2d, lat2d = np.meshgrid(lon, lat)

    # =========================
    # nearest 찾기
    # =========================
    def find_nearest(x, y):
        d2 = (Z_all[:, 0] - x) ** 2 + (Z_all[:, 1] - y) ** 2
        return np.argmin(d2)

    # =========================
    # raw data 읽기
    # =========================
    def load_raw(i):
        one_file = file_list[i]
        one_img = np.load(one_file)

        t = one_img[:, :, 0]
        z = one_img[:, :, 1]
        u = one_img[:, :, 2]
        v = one_img[:, :, 3]

        return t, z, u, v

    # =========================
    # 상태
    # =========================
    temp_levels = np.arange(-30, 31, 3)   # 고정
    state = {
        "drag": False,
        "idx": None,
        "cbar": None,
        "cax": None, #fig.add_axes([0.90, 0.15, 0.02, 0.7]),
        "path_indices": [],
    }

    # =========================
    # figure
    # =========================
    fig = plt.figure(1, figsize=(12, 6))
    ax_feat = fig.add_subplot(121)
    ax_map = fig.add_subplot(122)
    
    # fig.subplots_adjust(right=0.88)
    # ax_feat = fig.add_subplot(121)
    # ax_map = fig.add_subplot(122, projection=ccrs.PlateCarree())

    state["cax"] = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    sc = ax_feat.scatter(
        Z_all[:, 0],
        Z_all[:, 1],
        s=4,
        c="gray",
        alpha=0.5
    )

    selected_sc = ax_feat.scatter(
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2
    )

    path_sc = ax_feat.scatter(
        [],
        [],
        s=20,
        c="orange",
        alpha=0.9,
        zorder=4
    )

    path_line, = ax_feat.plot(
        [],
        [],
        color="orange",
        linewidth=1.5,
        alpha=0.9,
        zorder=3
    )

    ax_feat.set_title("Feature space")
    ax_map.set_title("850 hPa map")

    def idx_to_datestr(i, start=datetime(2020, 1, 1, 0, 0), step_hours=6):
        dt = start + timedelta(hours=int(i) * int(step_hours))
        return dt.strftime("%Y-%m-%d %H:%M")

    
    def draw_map(i):
        t, z, u, v = load_raw(i)
        ax_map.cla()
        #draw_base_map(ax_map, lon, lat)    

        cf_last = _draw_850hpa_on_ax(
            ax_map,
            lon2d,
            lat2d,
            t,
            z,
            u,
            v,
            use_theta=False,
            barb_skip=8,
            title=f"idx = {idx_to_datestr(i)}",
            shaded_levels=temp_levels,
        )
        # colorbar는 처음 한 번만 생성
        if (cf_last is not None) and (state["cbar"] is None):
            state["cbar"] = fig.colorbar(cf_last, cax=state["cax"])
            state["cbar"].set_label("Temperature (°C)")

    def update_path_plot():
        if len(state["path_indices"]) == 0:
            path_sc.set_offsets(np.empty((0, 2)))
            path_line.set_data([], [])
            return

        path_xy = Z_all[state["path_indices"]]

        path_sc.set_offsets(path_xy)
        path_line.set_data(path_xy[:, 0], path_xy[:, 1])

    # =========================
    # mouse events
    # =========================

    def on_press(event):
        if event.inaxes != ax_feat:
            return

        if event.button != 1:
            return

        state["drag"] = True
        # 새 드래그 시작 -> 이전 path 초기화
        state["path_indices"] = []
        update_path_plot()

        update_mouse(event)

    def on_release(event):
        if event.button == 1:
            state["drag"] = False
            
            # path 지우기
            state["path_indices"] = []
            update_path_plot()

            fig.canvas.draw_idle()

    def on_move(event):
        if not state["drag"]:
            return
        if event.inaxes != ax_feat:
            return
        update_mouse(event)

    def update_mouse(event):
        if event.xdata is None  or event.ydata is None:
            return
        x = event.xdata
        y = event.ydata

        i = find_nearest(x, y)
        if state["idx"] == i:
            return

        state["idx"] = i
        # path에 새 점 추가
        state["path_indices"].append(int(i))
        # 표시 점 업데이트
        selected_sc.set_offsets(Z_all[i])
        # path 표시 업데이트
        update_path_plot()
        # 지도 업데이트
        draw_map(i)

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["t"]:
            start_i = state["idx"]
            navigate_on_feats_temporal(Z_all, file_list, start_i)              
        

    # =========================
    # connect
    # =========================
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)


    plt.show()

def navigate_on_feats_and_replay(Z_all, file_list):    
    # =========================
    # 고정 grid
    # =========================
    lon = np.arange(105.25, 147, 0.25)   # 180
    lat = np.arange(58.75, 16, -0.25)    # 180
    lon2d, lat2d = np.meshgrid(lon, lat)

    # =========================
    # nearest 찾기
    # =========================
    def find_nearest(x, y):
        d2 = (Z_all[:, 0] - x) ** 2 + (Z_all[:, 1] - y) ** 2
        return np.argmin(d2)

    # =========================
    # raw data 읽기
    # =========================
    def load_raw(i):
        one_file = file_list[i]
        one_img = np.load(one_file)

        t = one_img[:, :, 0]
        z = one_img[:, :, 1]
        u = one_img[:, :, 2]
        v = one_img[:, :, 3]

        return t, z, u, v

    # =========================
    # 상태
    # =========================
    temp_levels = np.arange(-30, 31, 3)   # 고정
    state = {
        "drag": False,
        "idx": None,
        "cbar": None,
        "cax": None, #fig.add_axes([0.90, 0.15, 0.02, 0.7]),
        "path_indices": [],
        "last_path_indices": [],   # 마지막으로 릴리즈된 path

        "is_replaying": False,
        "replay_pos": 0,
        "replay_timer": None,
        "replay_token": 0,         # 새 드래그가 시작되면 이전 replay 무효화

        "replay_interval_ms": 1000,   # ← 추가
    }

    # =========================
    # figure
    # =========================
    fig = plt.figure(1, figsize=(12, 6))
    ax_feat = fig.add_subplot(121)
    ax_map = fig.add_subplot(122)
    
    # fig.subplots_adjust(right=0.88)
    # ax_feat = fig.add_subplot(121)
    # ax_map = fig.add_subplot(122, projection=ccrs.PlateCarree())

    state["cax"] = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    sc = ax_feat.scatter(
        Z_all[:, 0],
        Z_all[:, 1],
        s=4,
        c="gray",
        alpha=0.5
    )

    selected_sc = ax_feat.scatter(
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2
    )

    path_sc = ax_feat.scatter(
        [],
        [],
        s=20,
        c="orange",
        alpha=0.9,
        zorder=4
    )

    path_line, = ax_feat.plot(
        [],
        [],
        color="orange",
        linewidth=1.5,
        alpha=0.9,
        zorder=3
    )

    replay_sc = ax_feat.scatter(
        [],
        [],
        s=140,
        c="cyan",
        edgecolors="black",
        linewidths=1.5,
        zorder=6
    )

    ax_feat.set_title("Feature space")
    ax_map.set_title("850 hPa map")

    def idx_to_datestr(i, start=datetime(2020, 1, 1, 0, 0), step_hours=6):
        dt = start + timedelta(hours=int(i) * int(step_hours))
        return dt.strftime("%Y-%m-%d %H:%M")

    
    def draw_map(i, title_prefix=""):
        t, z, u, v = load_raw(i)
        ax_map.cla()

        title_str = idx_to_datestr(i)
        if title_prefix:
            title_str = f"{title_prefix} | {title_str}"

        cf_last = _draw_850hpa_on_ax(
            ax_map,
            lon2d,
            lat2d,
            t,
            z,
            u,
            v,
            use_theta=False,
            barb_skip=8,
            title=title_str,
            shaded_levels=temp_levels,
        )

        # colorbar는 처음 한 번만 생성
        if (cf_last is not None) and (state["cbar"] is None):
            state["cbar"] = fig.colorbar(cf_last, cax=state["cax"])
            state["cbar"].set_label("Temperature (°C)")

    def update_path_plot():
        if len(state["path_indices"]) == 0:
            path_sc.set_offsets(np.empty((0, 2)))
            path_line.set_data([], [])
            return

        path_xy = Z_all[state["path_indices"]]

        path_sc.set_offsets(path_xy)
        path_line.set_data(path_xy[:, 0], path_xy[:, 1])

    def update_replay_marker(i=None):
        if i is None:
            replay_sc.set_offsets(np.empty((0, 2)))
        else:
            replay_sc.set_offsets(Z_all[int(i):int(i)+1])

    def stop_replay():
        state["is_replaying"] = False
        state["replay_pos"] = 0
        state["replay_token"] += 1

        if state["replay_timer"] is not None:
            try:
                state["replay_timer"].stop()
            except Exception:
                pass
            state["replay_timer"] = None

        update_replay_marker(None)

    def start_replay(interval_ms=None):
        if interval_ms is None:
            interval_ms = state["replay_interval_ms"]

        indices = state["last_path_indices"]

        if indices is None or len(indices) == 0:
            return

        stop_replay()

        state["is_replaying"] = True
        state["replay_pos"] = 0
        my_token = state["replay_token"]

        timer = fig.canvas.new_timer(interval=interval_ms)
        state["replay_timer"] = timer

        def _tick():
            # 새 드래그가 시작되었거나 replay가 중단되었으면 종료
            if (not state["is_replaying"]) or (my_token != state["replay_token"]):
                try:
                    timer.stop()
                except Exception:
                    pass
                return

            indices_local = state["last_path_indices"]
            if len(indices_local) == 0:
                try:
                    timer.stop()
                except Exception:
                    pass
                return

            k = state["replay_pos"] % len(indices_local)
            i = indices_local[k]

            # feature space에서 현재 재생 위치 표시
            update_replay_marker(i)

            # map 재생
            draw_map(i, title_prefix=f"Replay {k+1}/{len(indices_local)}")

            fig.canvas.draw_idle()

            state["replay_pos"] += 1

        timer.add_callback(_tick)
        timer.start()

    def change_replay_speed(delta):
        # delta > 0 → 느리게
        # delta < 0 → 빠르게

        state["replay_interval_ms"] += delta

        if state["replay_interval_ms"] < 30:
            state["replay_interval_ms"] = 30

        if state["replay_interval_ms"] > 2000:
            state["replay_interval_ms"] = 2000

        print("interval =", state["replay_interval_ms"], "ms")

        # 재생 중이면 재시작
        if state["is_replaying"]:
            start_replay()

    # =========================
    # mouse events
    # =========================

    def on_press(event):
        if event.inaxes != ax_feat:
            return

        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        # 이전 replay 중단
        stop_replay()

        state["drag"] = True
        # 새 드래그 시작 -> 이전 path 초기화
        state["path_indices"] = []

        update_path_plot()
        update_mouse(event)

    def on_release(event):
        if event.button != 1:
            return

        if state["drag"] is not True:
            return

        state["drag"] = False
        state["last_path_indices"] = state["path_indices"][:]

        fig.canvas.draw_idle()
        #start_replay()

    def on_move(event):
        if state["drag"] is not True:
            return
        if event.inaxes != ax_feat:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        update_mouse(event)

    def update_mouse(event):
        if event.xdata is None  or event.ydata is None:
            return
        x = event.xdata
        y = event.ydata

        i = find_nearest(x, y)
        if state["idx"] == i:
            return

        state["idx"] = i
        if len(state["path_indices"]) == 0 or state["path_indices"][-1] != i:
            state["path_indices"].append(i)

        # # path에 새 점 추가
        # state["path_indices"].append(int(i))
        # 표시 점 업데이트
        selected_sc.set_offsets(Z_all[i])
        # path 표시 업데이트
        update_path_plot()
        # 지도 업데이트
        draw_map(i, title_prefix="Live")

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "t":
            start_i = state["idx"]
            navigate_on_feats_temporal(Z_all, file_list, start_i)

        elif event.key == "escape":
            stop_replay()
            fig.canvas.draw_idle()

        elif event.key == "+":
            change_replay_speed(-100)

        elif event.key == "-":
            change_replay_speed(+100)

        elif event.key == "r":
            start_replay()
        

    # =========================
    # connect
    # =========================
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)


    plt.show()