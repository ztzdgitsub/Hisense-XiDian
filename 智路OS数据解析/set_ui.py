import streamlit as st
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# 配置页面
st.set_page_config(
    page_title="基础配置管理 - 人车非目标检测系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class NodeManagementSystem:
    def __init__(self):
        self.nodes = {}
        self.cameras = {}
        self.intersections = {}

    def add_node(self, node_data: Dict):
        """添加边缘节点"""
        node_id = node_data.get('id', str(uuid.uuid4()))
        node_data['id'] = node_id
        node_data['created_at'] = datetime.now().isoformat()
        node_data['status'] = 'online'
        self.nodes[node_id] = node_data
        return node_id

    def update_node(self, node_id: str, updates: Dict):
        """更新节点信息"""
        if node_id in self.nodes:
            self.nodes[node_id].update(updates)
            self.nodes[node_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False

    def delete_node(self, node_id: str):
        """删除节点"""
        if node_id in self.nodes:
            # 同时删除该节点关联的摄像头
            for cam_id, camera in list(self.cameras.items()):
                if camera.get('node_id') == node_id:
                    del self.cameras[cam_id]
            del self.nodes[node_id]
            return True
        return False

    def add_camera(self, camera_data: Dict):
        """添加摄像头"""
        camera_id = camera_data.get('id', str(uuid.uuid4()))
        camera_data['id'] = camera_id
        camera_data['created_at'] = datetime.now().isoformat()
        self.cameras[camera_id] = camera_data
        return camera_id

    def update_camera(self, camera_id: str, updates: Dict):
        """更新摄像头信息"""
        if camera_id in self.cameras:
            self.cameras[camera_id].update(updates)
            self.cameras[camera_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False

    def delete_camera(self, camera_id: str):
        """删除摄像头"""
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            return True
        return False

    def add_intersection(self, intersection_data: Dict):
        """添加路口"""
        intersection_id = intersection_data.get('id', str(uuid.uuid4()))
        intersection_data['id'] = intersection_id
        intersection_data['created_at'] = datetime.now().isoformat()
        self.intersections[intersection_id] = intersection_data
        return intersection_id

    def get_node_cameras(self, node_id: str) -> List[Dict]:
        """获取节点关联的摄像头"""
        return [cam for cam in self.cameras.values() if cam.get('node_id') == node_id]

    def get_slave_nodes(self, master_node_id: str) -> List[Dict]:
        """获取从节点"""
        return [node for node in self.nodes.values() if node.get('master_node_id') == master_node_id]

    def export_configuration(self) -> Dict:
        """导出完整配置"""
        return {
            "export_time": datetime.now().isoformat(),
            "version": "1.0",
            "nodes": self.nodes,
            "cameras": self.cameras,
            "intersections": self.intersections
        }


def initialize_session_state():
    """初始化会话状态"""
    if 'node_system' not in st.session_state:
        st.session_state.node_system = NodeManagementSystem()
        # 添加示例数据
        _add_sample_data()

    if 'editing_node' not in st.session_state:
        st.session_state.editing_node = None

    if 'editing_camera' not in st.session_state:
        st.session_state.editing_camera = None

    if 'editing_intersection' not in st.session_state:
        st.session_state.editing_intersection = None

    # 初始化临时区域数据
    if 'temp_areas' not in st.session_state:
        st.session_state.temp_areas = []

    # 初始化区域管理状态
    if 'area_management' not in st.session_state:
        st.session_state.area_management = {
            'new_area_input': '',
            'areas_to_delete': []
        }


def _add_sample_data():
    """添加示例数据"""
    system = st.session_state.node_system

    # 添加主节点
    master_node_id = system.add_node({
        "name": "路口A-中心节点",
        "ip_address": "192.168.1.100",
        "model": "Jetson AGX Orin 64GB",
        "version": "v2.1.0",
        "location": "路口A-东北角灯杆",
        "is_master": True,
        "description": "主处理节点，负责数据汇总"
    })

    # 添加从节点
    slave_node_id = system.add_node({
        "name": "路口A-南向节点",
        "ip_address": "192.168.1.101",
        "model": "Jetson AGX Orin 32GB",
        "version": "v2.1.0",
        "location": "路口A-南向灯杆",
        "is_master": False,
        "master_node_id": master_node_id,
        "description": "南向视频流处理"
    })

    # 添加摄像头
    system.add_camera({
        "name": "北向主相机",
        "rtsp_url": "rtsp://192.168.1.201:554/stream1",
        "ip_address": "192.168.1.201",
        "port": 554,
        "username": "admin",
        "password": "******",
        "encoding": "H.264",
        "resolution": "1920x1080",
        "node_id": master_node_id,
        "status": "online",
        "video_quality": 95
    })

    system.add_camera({
        "name": "南向辅相机",
        "rtsp_url": "rtsp://192.168.1.202:554/stream1",
        "ip_address": "192.168.1.202",
        "port": 554,
        "username": "admin",
        "password": "******",
        "encoding": "H.265",
        "resolution": "2560x1440",
        "node_id": slave_node_id,
        "status": "online",
        "video_quality": 92
    })

    # 添加路口
    system.add_intersection({
        "name": "路口A",
        "location": "人民路与解放路交叉口",
        "description": "主要交通路口，人车流量大",
        "nodes": [master_node_id, slave_node_id],
        "cameras": ["cam_1", "cam_2"],
        "areas": ["机动车道", "非机动车道", "人行横道"]
    })


def create_topology_graph(system):
    """创建拓扑关系图"""
    G = nx.DiGraph()

    # 添加节点
    for node_id, node in system.nodes.items():
        node_type = "主节点" if node.get('is_master') else "从节点"
        G.add_node(
            node_id,
            label=f"{node['name']}\n{node['ip_address']}\n{node_type}",
            type=node_type
        )

    # 添加主从关系边
    for node_id, node in system.nodes.items():
        if not node.get('is_master') and node.get('master_node_id'):
            G.add_edge(node['master_node_id'], node_id, relationship="主从")

    # 添加摄像头关系
    for camera_id, camera in system.cameras.items():
        node_id = camera.get('node_id')
        if node_id and node_id in system.nodes:
            G.add_node(
                camera_id,
                label=f"{camera['name']}\n摄像头",
                type="摄像头"
            )
            G.add_edge(node_id, camera_id, relationship="数据处理")

    return G


def draw_topology_chart(system):
    """绘制拓扑图"""
    G = create_topology_graph(system)

    if len(G.nodes) == 0:
        st.info("暂无节点数据，请先添加节点和摄像头")
        return

    # 使用networkx的布局算法
    pos = nx.spring_layout(G, k=3, iterations=50)

    # 提取节点位置
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = G.nodes[node]
        node_text.append(node_data.get('label', node))

        # 根据节点类型设置颜色
        node_type = node_data.get('type', '未知')
        if node_type == "主节点":
            node_color.append('#FF6B6B')  # 红色
        elif node_type == "从节点":
            node_color.append('#4ECDC4')  # 青色
        else:
            node_color.append('#45B7D1')  # 蓝色

    # 创建节点轨迹
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            color=node_color,
            size=40,
            line=dict(width=2, color='darkblue')
        )
    )

    # 创建边轨迹
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # 创建图表
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='系统拓扑关系图',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))

    st.plotly_chart(fig, use_container_width=True)


def main():
    # 初始化
    initialize_session_state()

    st.title("⚙️ 基础配置管理 - 人车非目标检测系统")

    # 侧边栏 - 快速操作
    with st.sidebar:
        st.header("快速操作")

        # 快速添加
        if st.button("➕ 快速添加节点", use_container_width=True):
            st.session_state.editing_node = "new"

        if st.button("📷 快速添加摄像头", use_container_width=True):
            st.session_state.editing_camera = "new"

        if st.button("🛣️ 快速添加路口", use_container_width=True):
            st.session_state.editing_intersection = "new"

        st.divider()

        # 系统操作
        if st.button("💾 导出配置", use_container_width=True):
            config_data = st.session_state.node_system.export_configuration()
            st.download_button(
                label="下载配置文件",
                data=json.dumps(config_data, indent=2, ensure_ascii=False),
                file_name=f"system_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        if st.button("🔄 导入配置", use_container_width=True):
            st.info("配置导入功能开发中...")

        if st.button("🧹 清空所有数据", use_container_width=True, type="secondary"):
            if st.checkbox("确认清空所有数据？此操作不可恢复！"):
                st.session_state.node_system = NodeManagementSystem()
                st.rerun()

    # 主内容区 - 标签页布局
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 系统总览",
        "🖥️ 节点管理",
        "📷 摄像头管理",
        "🛣️ 路口管理"
    ])

    with tab1:
        st.header("系统总览")

        # 系统统计卡片
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_nodes = len(st.session_state.node_system.nodes)
            master_nodes = len([n for n in st.session_state.node_system.nodes.values() if n.get('is_master')])
            st.metric("边缘节点", f"{total_nodes} 个", f"主节点: {master_nodes} 个")

        with col2:
            total_cameras = len(st.session_state.node_system.cameras)
            online_cameras = len(
                [c for c in st.session_state.node_system.cameras.values() if c.get('status') == 'online'])
            st.metric("摄像头", f"{total_cameras} 个", f"在线: {online_cameras} 个")

        with col3:
            total_intersections = len(st.session_state.node_system.intersections)
            st.metric("路口", f"{total_intersections} 个", "监控点位")

        with col4:
            system_status = "正常" if total_nodes > 0 and online_cameras > 0 else "异常"
            status_color = {"正常": "normal", "异常": "off"}
            st.metric("系统状态", system_status, "运行中")

        # 拓扑图
        st.subheader("系统拓扑图")
        draw_topology_chart(st.session_state.node_system)

        # 节点状态表格
        st.subheader("节点状态监控")
        if st.session_state.node_system.nodes:
            node_data = []
            for node_id, node in st.session_state.node_system.nodes.items():
                cameras = st.session_state.node_system.get_node_cameras(node_id)
                node_data.append({
                    "节点名称": node.get('name', '未知'),
                    "IP地址": node.get('ip_address', '未知'),
                    "型号": node.get('model', '未知'),
                    "位置": node.get('location', '未知'),
                    "节点类型": "主节点" if node.get('is_master') else "从节点",
                    "关联摄像头": len(cameras),
                    "状态": node.get('status', 'unknown')
                })

            df = pd.DataFrame(node_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无节点数据")

    with tab2:
        st.header("边缘节点管理")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 节点列表
            st.subheader("节点列表")
            if st.session_state.node_system.nodes:
                for node_id, node in st.session_state.node_system.nodes.items():
                    with st.expander(f"🖥️ {node.get('name', '未知节点')} - {node.get('ip_address', '未知IP')}",
                                     expanded=False):
                        col_a, col_b, col_c = st.columns([3, 1, 1])

                        with col_a:
                            st.write(f"**型号:** {node.get('model', '未知')}")
                            st.write(f"**版本:** {node.get('version', '未知')}")
                            st.write(f"**位置:** {node.get('location', '未知')}")
                            st.write(f"**类型:** {'主节点' if node.get('is_master') else '从节点'}")

                            if not node.get('is_master') and node.get('master_node_id'):
                                master_node = st.session_state.node_system.nodes.get(node['master_node_id'])
                                if master_node:
                                    st.write(f"**主节点:** {master_node.get('name')}")

                            st.write(f"**状态:** {node.get('status', 'unknown')}")

                            # 关联摄像头
                            cameras = st.session_state.node_system.get_node_cameras(node_id)
                            if cameras:
                                st.write(f"**关联摄像头:** {len(cameras)} 个")
                                for cam in cameras:
                                    st.write(f"  - {cam.get('name')} ({cam.get('status', 'unknown')})")

                        with col_b:
                            if st.button("编辑", key=f"edit_node_{node_id}"):
                                st.session_state.editing_node = node_id

                        with col_c:
                            if st.button("删除", key=f"delete_node_{node_id}"):
                                if st.session_state.node_system.delete_node(node_id):
                                    st.success("节点删除成功！")
                                    st.rerun()
            else:
                st.info("暂无节点数据")

        with col2:
            # 节点编辑/添加表单
            st.subheader("节点配置")

            if st.session_state.editing_node:
                if st.session_state.editing_node == "new":
                    node_data = {}
                    form_title = "添加新节点"
                else:
                    node_data = st.session_state.node_system.nodes.get(st.session_state.editing_node, {})
                    form_title = "编辑节点"

                with st.form(f"node_form_{st.session_state.editing_node}"):
                    st.write(f"**{form_title}**")

                    name = st.text_input("节点名称", value=node_data.get('name', ''))
                    ip_address = st.text_input("IP地址", value=node_data.get('ip_address', ''))
                    model = st.selectbox(
                        "硬件型号",
                        ["Jetson AGX Orin 64GB", "Jetson AGX Orin 32GB", "Jetson AGX Orin 16GB", "其他型号"],
                        index=0 if not node_data else ["Jetson AGX Orin 64GB", "Jetson AGX Orin 32GB",
                                                       "Jetson AGX Orin 16GB", "其他型号"].index(
                            node_data.get('model', 'Jetson AGX Orin 64GB'))
                    )
                    version = st.text_input("软件版本", value=node_data.get('version', 'v2.1.0'))
                    location = st.text_input("安装位置", value=node_data.get('location', ''))

                    is_master = st.checkbox("设为主节点", value=node_data.get('is_master', False))

                    # 如果不是主节点，可以选择主节点
                    master_node_options = [nid for nid, n in st.session_state.node_system.nodes.items() if
                                           n.get('is_master')]
                    if not is_master and master_node_options:
                        current_master = node_data.get('master_node_id')
                        master_node_id = st.selectbox(
                            "选择主节点",
                            options=master_node_options,
                            format_func=lambda x: st.session_state.node_system.nodes[x].get('name'),
                            index=master_node_options.index(
                                current_master) if current_master in master_node_options else 0
                        )
                    else:
                        master_node_id = None

                    description = st.text_area("描述信息", value=node_data.get('description', ''))

                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        if st.form_submit_button("保存配置", use_container_width=True):
                            if name and ip_address:
                                new_node_data = {
                                    "name": name,
                                    "ip_address": ip_address,
                                    "model": model,
                                    "version": version,
                                    "location": location,
                                    "is_master": is_master,
                                    "master_node_id": master_node_id if not is_master else None,
                                    "description": description
                                }

                                if st.session_state.editing_node == "new":
                                    st.session_state.node_system.add_node(new_node_data)
                                    st.success("节点添加成功！")
                                else:
                                    st.session_state.node_system.update_node(st.session_state.editing_node,
                                                                             new_node_data)
                                    st.success("节点更新成功！")

                                st.session_state.editing_node = None
                                st.rerun()
                            else:
                                st.error("请填写节点名称和IP地址")

                    with col_cancel:
                        if st.form_submit_button("取消", use_container_width=True, type="secondary"):
                            st.session_state.editing_node = None
                            st.rerun()

            else:
                st.info("选择左侧节点进行编辑，或点击'添加新节点'")

                # 节点健康状态
                st.subheader("节点状态")
                for node_id, node in st.session_state.node_system.nodes.items():
                    status = node.get('status', 'unknown')
                    status_color = {
                        'online': '🟢',
                        'offline': '🔴',
                        'unknown': '⚫'
                    }
                    st.write(f"{status_color.get(status, '⚫')} {node.get('name')}: {status}")

    with tab3:
        st.header("摄像头管理")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 摄像头列表
            st.subheader("摄像头列表")
            if st.session_state.node_system.cameras:
                for camera_id, camera in st.session_state.node_system.cameras.items():
                    with st.expander(f"📷 {camera.get('name', '未知摄像头')} - {camera.get('ip_address', '未知IP')}",
                                     expanded=False):
                        col_a, col_b, col_c = st.columns([3, 1, 1])

                        with col_a:
                            st.write(f"**RTSP地址:** {camera.get('rtsp_url', '未知')}")
                            st.write(f"**编码格式:** {camera.get('encoding', '未知')}")
                            st.write(f"**分辨率:** {camera.get('resolution', '未知')}")
                            st.write(f"**视频质量:** {camera.get('video_quality', '未知')}")
                            st.write(f"**状态:** {camera.get('status', 'unknown')}")

                            # 关联节点
                            node_id = camera.get('node_id')
                            if node_id:
                                node = st.session_state.node_system.nodes.get(node_id)
                                if node:
                                    st.write(f"**处理节点:** {node.get('name')}")

                        with col_b:
                            if st.button("编辑", key=f"edit_camera_{camera_id}"):
                                st.session_state.editing_camera = camera_id

                        with col_c:
                            if st.button("删除", key=f"delete_camera_{camera_id}"):
                                if st.session_state.node_system.delete_camera(camera_id):
                                    st.success("摄像头删除成功！")
                                    st.rerun()
            else:
                st.info("暂无摄像头数据")

        with col2:
            # 摄像头编辑/添加表单
            st.subheader("摄像头配置")

            if st.session_state.editing_camera:
                if st.session_state.editing_camera == "new":
                    camera_data = {}
                    form_title = "添加新摄像头"
                else:
                    camera_data = st.session_state.node_system.cameras.get(st.session_state.editing_camera, {})
                    form_title = "编辑摄像头"

                with st.form(f"camera_form_{st.session_state.editing_camera}"):
                    st.write(f"**{form_title}**")

                    name = st.text_input("摄像头名称", value=camera_data.get('name', ''))
                    rtsp_url = st.text_input("RTSP流地址", value=camera_data.get('rtsp_url', ''))
                    ip_address = st.text_input("IP地址", value=camera_data.get('ip_address', ''))
                    port = st.number_input("端口", min_value=1, max_value=65535, value=camera_data.get('port', 554))
                    username = st.text_input("用户名", value=camera_data.get('username', 'admin'))
                    password = st.text_input("密码", type="password", value=camera_data.get('password', ''))

                    encoding = st.selectbox(
                        "视频编码",
                        ["H.264", "H.265"],
                        index=0 if not camera_data else ["H.264", "H.265"].index(camera_data.get('encoding', 'H.264'))
                    )

                    resolution = st.selectbox(
                        "分辨率",
                        ["1920x1080", "2560x1440", "3840x2160", "1280x720"],
                        index=0 if not camera_data else ["1920x1080", "2560x1440", "3840x2160", "1280x720"].index(
                            camera_data.get('resolution', '1920x1080'))
                    )

                    # 选择处理节点
                    node_options = list(st.session_state.node_system.nodes.keys())
                    if node_options:
                        current_node = camera_data.get('node_id')
                        node_id = st.selectbox(
                            "处理节点",
                            options=node_options,
                            format_func=lambda x: st.session_state.node_system.nodes[x].get('name'),
                            index=node_options.index(current_node) if current_node in node_options else 0
                        )
                    else:
                        st.warning("请先添加节点")
                        node_id = None

                    status = st.selectbox(
                        "状态",
                        ["online", "offline", "maintenance"],
                        index=0 if not camera_data else ["online", "offline", "maintenance"].index(
                            camera_data.get('status', 'online'))
                    )

                    video_quality = st.slider("视频质量评分", 0, 100, value=camera_data.get('video_quality', 90))

                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        if st.form_submit_button("保存配置", use_container_width=True):
                            if name and rtsp_url and ip_address and node_id:
                                new_camera_data = {
                                    "name": name,
                                    "rtsp_url": rtsp_url,
                                    "ip_address": ip_address,
                                    "port": port,
                                    "username": username,
                                    "password": password,
                                    "encoding": encoding,
                                    "resolution": resolution,
                                    "node_id": node_id,
                                    "status": status,
                                    "video_quality": video_quality
                                }

                                if st.session_state.editing_camera == "new":
                                    st.session_state.node_system.add_camera(new_camera_data)
                                    st.success("摄像头添加成功！")
                                else:
                                    st.session_state.node_system.update_camera(st.session_state.editing_camera,
                                                                               new_camera_data)
                                    st.success("摄像头更新成功！")

                                st.session_state.editing_camera = None
                                st.rerun()
                            else:
                                st.error("请填写必填字段")

                    with col_cancel:
                        if st.form_submit_button("取消", use_container_width=True, type="secondary"):
                            st.session_state.editing_camera = None
                            st.rerun()

            else:
                st.info("选择左侧摄像头进行编辑，或点击'快速添加摄像头'")

    with tab4:
        st.header("路口管理")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 路口列表
            st.subheader("路口列表")
            if st.session_state.node_system.intersections:
                for intersection_id, intersection in st.session_state.node_system.intersections.items():
                    with st.expander(f"🛣️ {intersection.get('name', '未知路口')}", expanded=False):
                        col_a, col_b = st.columns([4, 1])

                        with col_a:
                            st.write(f"**位置:** {intersection.get('location', '未知')}")
                            st.write(f"**描述:** {intersection.get('description', '无')}")

                            # 关联节点
                            node_ids = intersection.get('nodes', [])
                            if node_ids:
                                st.write("**关联节点:**")
                                for node_id in node_ids:
                                    node = st.session_state.node_system.nodes.get(node_id)
                                    if node:
                                        st.write(f"  - {node.get('name')} ({node.get('ip_address')})")

                            # 显示关联摄像头
                            camera_ids = intersection.get('cameras', [])
                            if camera_ids:
                                st.write("**关联摄像头:**")
                                for cam_id in camera_ids:
                                    camera = st.session_state.node_system.cameras.get(cam_id)
                                    if camera:
                                        st.write(f"  - {camera.get('name')}")

                            # 显示关联区域
                            areas = intersection.get('areas', [])
                            if areas:
                                st.write("**关联区域:**")
                                for area in areas:
                                    st.write(f"  - {area}")

                        with col_b:
                            if st.button("删除", key=f"delete_intersection_{intersection_id}"):
                                if st.session_state.node_system.intersections.get(intersection_id):
                                    del st.session_state.node_system.intersections[intersection_id]
                                    st.success("路口删除成功！")
                                    st.rerun()
            else:
                st.info("暂未配置路口信息")

        with col2:
            # 路口编辑/添加表单（加框显示）
            with st.container():
                # 使用自定义CSS为容器添加边框
                st.markdown(
                    """
                    <style>
                    .bordered-container {
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px 0;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("路口配置")

                if st.session_state.editing_intersection:
                    if st.session_state.editing_intersection == "new":
                        intersection_data = {}
                        form_title = "添加新路口"
                        # 初始化临时区域数据
                        if 'temp_areas' not in st.session_state:
                            st.session_state.temp_areas = []
                    else:
                        intersection_data = st.session_state.node_system.intersections.get(
                            st.session_state.editing_intersection, {})
                        form_title = "编辑路口"
                        # 初始化临时区域数据
                        st.session_state.temp_areas = intersection_data.get('areas', [])

                    # 表单部分
                    with st.form(f"intersection_form_{st.session_state.editing_intersection}"):
                        st.write(f"**{form_title}**")

                        name = st.text_input("路口名称", value=intersection_data.get('name', ''))
                        location = st.text_input("具体位置", value=intersection_data.get('location', ''))
                        description = st.text_area("路口描述", value=intersection_data.get('description', ''))

                        # 选择关联节点
                        node_options = list(st.session_state.node_system.nodes.keys())
                        current_nodes = intersection_data.get('nodes', [])
                        selected_nodes = st.multiselect(
                            "关联节点",
                            options=node_options,
                            default=current_nodes,
                            format_func=lambda x: st.session_state.node_system.nodes[x].get('name')
                        )

                        # 多选关联摄像头
                        available_cameras = list(st.session_state.node_system.cameras.keys())
                        current_cameras = intersection_data.get('cameras', [])
                        selected_cameras = st.multiselect(
                            "关联摄像头（可多选）",
                            options=available_cameras,
                            default=current_cameras,
                            format_func=lambda x: st.session_state.node_system.cameras[x].get('name')
                        )

                        col_submit, col_cancel = st.columns(2)
                        with col_submit:
                            if st.form_submit_button("保存配置", use_container_width=True):
                                if name and location:
                                    # 获取临时存储的区域数据
                                    final_areas = st.session_state.get('temp_areas', [])

                                    new_intersection_data = {
                                        "name": name,
                                        "location": location,
                                        "description": description,
                                        "nodes": selected_nodes,
                                        "cameras": selected_cameras,
                                        "areas": final_areas  # 使用最终的区域数据
                                    }

                                    if st.session_state.editing_intersection == "new":
                                        st.session_state.node_system.add_intersection(new_intersection_data)
                                        st.success("路口添加成功！")
                                    else:
                                        st.session_state.node_system.intersections[
                                            st.session_state.editing_intersection].update(new_intersection_data)
                                        st.success("路口更新成功！")

                                    # 清理临时数据
                                    if 'temp_areas' in st.session_state:
                                        del st.session_state.temp_areas

                                    st.session_state.editing_intersection = None
                                    st.rerun()
                                else:
                                    st.error("请填写路口名称和位置")

                        with col_cancel:
                            if st.form_submit_button("取消", use_container_width=True, type="secondary"):
                                # 清理临时数据
                                if 'temp_areas' in st.session_state:
                                    del st.session_state.temp_areas
                                st.session_state.editing_intersection = None
                                st.rerun()

            # 区域管理部分（在表单外部，不加框）
            if st.session_state.editing_intersection:
                st.markdown("---")  # 添加分隔线

                # 使用与"添加新路口"相同大小的标题
                st.write(f"**关联区域配置**")

                # 使用临时存储的区域数据
                areas = st.session_state.get('temp_areas', [])
                new_area = st.text_input("输入区域名称", key="new_area_input",
                                         placeholder="例如: 机动车道、非机动车道、人行横道等")

                # 添加区域按钮
                if st.button("添加区域", key="add_area_btn"):
                    if new_area and new_area.strip() and new_area not in areas:
                        areas.append(new_area.strip())
                        st.session_state.temp_areas = areas
                        st.rerun()

                # 显示已添加的区域列表
                if areas:
                    st.write("已添加的区域:")
                    for i, area in enumerate(areas):
                        # 使用水平布局
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.write(f"- {area}")
                        with cols[1]:
                            # 删除按钮
                            if st.button("删除", key=f"del_area_{i}"):
                                areas.pop(i)
                                st.session_state.temp_areas = areas
                                st.rerun()
            else:
                st.info("点击'快速添加路口'开始配置")


if __name__ == "__main__":
    main()