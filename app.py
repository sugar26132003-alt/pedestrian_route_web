from flask import (
    Flask,
    render_template,
    jsonify,
    send_file,
    request
)

from pathlib import Path

import pickle
import math
import io

import networkx as nx
import osmnx as ox

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


# ============================================================
# HEIC 지원
# ============================================================

register_heif_opener()


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# 기본 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "integrated_latest.pkl"
)

PHOTO_ROOT = (
    BASE_DIR
    / "photos"
)


# ============================================================
# 경로 계산 설정
# ============================================================

DISCOMFORT_FACTOR = 30

UNKNOWN_PENALTY = 2

WALKING_SPEED = 1.2


# ============================================================
# integrated_latest.pkl 불러오기
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\n❌ integrated_latest.pkl을 찾을 수 없습니다.\n"
        f"{DATA_PATH}\n"
    )


with open(
    DATA_PATH,
    "rb"
) as f:

    integrated_data = pickle.load(f)


result_df = (
    integrated_data[
        "result_df"
    ].copy()
)

G = (
    integrated_data[
        "G"
    ]
)

edges_wgs84 = (
    integrated_data[
        "edges_wgs84"
    ]
)


print("=" * 70)
print("✅ integrated_latest.pkl 로딩 완료")
print("사진 수:", len(result_df))
print("도로 Edge 수:", len(edges_wgs84))
print("=" * 70)


# ============================================================
# 안전한 값 변환
# ============================================================

def safe_value(value):

    try:

        if value is None:
            return None

        if (
            isinstance(value, float)
            and
            math.isnan(value)
        ):
            return None

        return value

    except Exception:

        return None


# ============================================================
# 불편도 색상
# ============================================================

def discomfort_color(score):

    score = float(score)

    if score <= 5:
        return "#16a34a"

    elif score <= 12:
        return "#f59e0b"

    else:
        return "#ef4444"


# ============================================================
# Edge별 최대 불편도
# ============================================================

edge_scores = (

    result_df

    .dropna(
        subset=[
            "edge_u",
            "edge_v",
            "edge_key"
        ]
    )

    .groupby(
        [
            "edge_u",
            "edge_v",
            "edge_key"
        ],
        as_index=False
    )

    .agg({
        "discomfort": "max",
        "filename": "count"
    })

    .rename(
        columns={
            "filename": "photo_count"
        }
    )

)


# ============================================================
# 모든 Edge 기본 cost
# ============================================================

for u, v, k, data in G.edges(
    keys=True,
    data=True
):

    distance = float(
        data.get(
            "length",
            1
        )
    )

    data["discomfort"] = 0

    data["has_photo"] = False

    data["route_score"] = (
        UNKNOWN_PENALTY
    )

    data["cost"] = (
        distance
        +
        UNKNOWN_PENALTY
        *
        DISCOMFORT_FACTOR
    )


# ============================================================
# 사진이 있는 Edge → 실제 불편도 적용
# ============================================================

for _, row in edge_scores.iterrows():

    u = row["edge_u"]
    v = row["edge_v"]
    k = row["edge_key"]

    score = float(
        row["discomfort"]
    )

    if G.has_edge(
        u,
        v,
        k
    ):

        distance = float(
            G[u][v][k].get(
                "length",
                1
            )
        )

        G[u][v][k][
            "has_photo"
        ] = True

        G[u][v][k][
            "discomfort"
        ] = score

        G[u][v][k][
            "route_score"
        ] = score

        G[u][v][k][
            "cost"
        ] = (
            distance
            +
            score
            *
            DISCOMFORT_FACTOR
        )


print(
    "✅ 경로 추천용 cost 설정 완료"
)


# ============================================================
# 메인 페이지
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# 로컬 사진 찾기
# ============================================================

def find_local_photo(
    area,
    filename
):

    area = str(
        area
    ).strip()

    filename = str(
        filename
    ).strip()


    direct_path = (
        PHOTO_ROOT
        / area
        / filename
    )


    if direct_path.exists():

        return direct_path


    area_dir = (
        PHOTO_ROOT
        / area
    )


    if area_dir.exists():

        matches = list(
            area_dir.rglob(
                filename
            )
        )

        if matches:

            return matches[0]


    return None


# ============================================================
# 사진 전송
# HEIC → JPEG 자동 변환
# ============================================================

@app.route(
    "/photo/<int:photo_id>"
)
def photo(photo_id):


    if (
        photo_id < 0
        or
        photo_id >= len(result_df)
    ):

        return (
            "사진을 찾을 수 없습니다.",
            404
        )


    row = (
        result_df.iloc[
            photo_id
        ]
    )


    area = row.get(
        "area",
        ""
    )

    filename = row.get(
        "filename",
        ""
    )


    image_path = (
        find_local_photo(
            area,
            filename
        )
    )


    if (
        image_path is None
        or
        not image_path.exists()
    ):

        return (
            "사진 파일이 없습니다.",
            404
        )


    suffix = (
        image_path
        .suffix
        .lower()
    )


    if suffix in [
        ".heic",
        ".heif"
    ]:

        try:

            with Image.open(
                image_path
            ) as img:

                img = (
                    ImageOps
                    .exif_transpose(
                        img
                    )
                )

                img = img.convert(
                    "RGB"
                )

                img.thumbnail(
                    (
                        1200,
                        1200
                    )
                )

                buffer = (
                    io.BytesIO()
                )

                img.save(
                    buffer,
                    format="JPEG",
                    quality=85
                )

                buffer.seek(0)


                return send_file(
                    buffer,
                    mimetype="image/jpeg",
                    download_name=(
                        image_path.stem
                        +
                        ".jpg"
                    )
                )


        except Exception as e:

            print(
                "❌ HEIC 변환 실패:",
                e
            )

            return (
                "HEIC 변환 실패",
                500
            )


    return send_file(
        image_path
    )


# ============================================================
# 두 Node 사이 최적 Edge
# ============================================================

def get_best_edge(
    graph,
    u,
    v,
    weight_name
):

    edge_dict = (
        graph.get_edge_data(
            u,
            v
        )
    )


    if not edge_dict:

        return None, None


    best_key = min(

        edge_dict,

        key=lambda key:

            float(

                edge_dict[
                    key
                ].get(
                    weight_name,
                    float("inf")
                )

            )

    )


    return (
        best_key,
        edge_dict[
            best_key
        ]
    )


# ============================================================
# Edge geometry → Leaflet 좌표
# ============================================================

def edge_coordinates(
    graph,
    u,
    v,
    edge_data
):

    geometry = (
        edge_data.get(
            "geometry"
        )
    )


    if geometry is not None:

        if (
            geometry.geom_type
            ==
            "LineString"
        ):

            return [

                [

                    [
                        lat,
                        lon
                    ]

                    for lon, lat
                    in geometry.coords

                ]

            ]


        elif (
            geometry.geom_type
            ==
            "MultiLineString"
        ):

            return [

                [

                    [
                        lat,
                        lon
                    ]

                    for lon, lat
                    in line.coords

                ]

                for line
                in geometry.geoms

            ]


    return [

        [

            [
                graph.nodes[u]["y"],
                graph.nodes[u]["x"]
            ],

            [
                graph.nodes[v]["y"],
                graph.nodes[v]["x"]
            ]

        ]

    ]


# ============================================================
# 경로 분석
# ============================================================

def build_route_data(
    graph,
    route,
    select_weight,
    route_type
):

    segments = []

    edge_ids = []

    total_distance = 0

    total_discomfort = 0

    unknown_edges = 0


    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        key, edge_data = (
            get_best_edge(
                graph,
                u,
                v,
                select_weight
            )
        )


        if edge_data is None:

            continue


        edge_ids.append(
            (
                u,
                v,
                key
            )
        )


        distance = float(
            edge_data.get(
                "length",
                0
            )
        )


        discomfort = float(
            edge_data.get(
                "discomfort",
                0
            )
        )


        route_score = float(
            edge_data.get(
                "route_score",
                UNKNOWN_PENALTY
            )
        )


        has_photo = bool(
            edge_data.get(
                "has_photo",
                False
            )
        )


        total_distance += (
            distance
        )

        total_discomfort += (
            discomfort
        )


        if not has_photo:

            unknown_edges += 1


        lines = (
            edge_coordinates(
                graph,
                u,
                v,
                edge_data
            )
        )


        for coordinates in lines:

            if (
                route_type
                ==
                "shortest"
            ):

                color = (
                    "#777777"
                )

            else:

                color = (
                    discomfort_color(
                        route_score
                    )
                )


            segments.append({

                "coordinates":
                    coordinates,

                "color":
                    color,

                "distance":
                    round(
                        distance,
                        2
                    ),

                "discomfort":
                    discomfort,

                "route_score":
                    route_score,

                "has_photo":
                    has_photo

            })


    time_minutes = (
        total_distance
        /
        WALKING_SPEED
        /
        60
    )


    return {

        "segments":
            segments,

        "edge_ids":
            edge_ids,

        "distance":
            round(
                total_distance,
                1
            ),

        "time":
            round(
                time_minutes,
                1
            ),

        "discomfort":
            round(
                total_discomfort,
                1
            ),

        "unknown_edges":
            int(
                unknown_edges
            )

    }


# ============================================================
# 추천 경로에 해당하는 사진만
# ============================================================

def get_route_photos(
    route_edge_ids
):

    route_edge_set = set(
        route_edge_ids
    )

    photos = []


    for position in range(
        len(result_df)
    ):

        row = (
            result_df.iloc[
                position
            ]
        )


        edge_u = safe_value(
            row.get(
                "edge_u"
            )
        )

        edge_v = safe_value(
            row.get(
                "edge_v"
            )
        )

        edge_key = safe_value(
            row.get(
                "edge_key"
            )
        )


        if (
            edge_u is None
            or
            edge_v is None
            or
            edge_key is None
        ):

            continue


        edge_id = (
            edge_u,
            edge_v,
            edge_key
        )


        if edge_id not in route_edge_set:

            continue


        lat = safe_value(
            row.get(
                "matched_lat"
            )
        )

        lon = safe_value(
            row.get(
                "matched_lon"
            )
        )


        if (
            lat is None
            or
            lon is None
        ):

            continue


        detections = {}


        for cls in [

            "crosswalk",
            "ramp",
            "curb",
            "obstacle",
            "stairs"

        ]:

            try:

                count = int(
                    row.get(
                        cls,
                        0
                    )
                )

            except Exception:

                count = 0


            if count > 0:

                detections[
                    cls
                ] = count


        discomfort = safe_value(
            row.get(
                "discomfort"
            )
        )


        if discomfort is None:

            discomfort = 0


        photos.append({

            "id":
                position,

            "filename":
                str(
                    row.get(
                        "filename",
                        ""
                    )
                ),

            "area":
                str(
                    row.get(
                        "area",
                        ""
                    )
                ),

            "lat":
                float(
                    lat
                ),

            "lon":
                float(
                    lon
                ),

            "discomfort":
                float(
                    discomfort
                ),

            "detections":
                detections,

            "image_url":
                f"/photo/{position}"

        })


    return photos


# ============================================================
# 경로 검색 API
# ============================================================

@app.route(
    "/api/route",
    methods=[
        "POST"
    ]
)
def route_api():


    try:

        body = (
            request.get_json(
                silent=True
            )
            or
            {}
        )


        start_query = str(
            body.get(
                "start",
                ""
            )
        ).strip()


        end_query = str(
            body.get(
                "end",
                ""
            )
        ).strip()


        if not start_query:

            return jsonify({
                "success": False,
                "message":
                    "출발지를 입력해주세요."
            }), 400


        if not end_query:

            return jsonify({
                "success": False,
                "message":
                    "도착지를 입력해주세요."
            }), 400


        start_search = (
            start_query
            +
            ", 세종특별자치시 조치원읍, 대한민국"
        )


        end_search = (
            end_query
            +
            ", 세종특별자치시 조치원읍, 대한민국"
        )


        print()
        print("=" * 70)

        print("🔎 경로 검색")

        print(
            "출발지:",
            start_query
        )

        print(
            "도착지:",
            end_query
        )


        # ====================================================
        # 장소명 → 좌표
        # ====================================================

        try:

            START_LAT, START_LON = (
                ox.geocode(
                    start_search
                )
            )

        except Exception:

            return jsonify({

                "success":
                    False,

                "message":
                    f"출발지 '{start_query}'를 찾지 못했습니다."

            }), 400


        try:

            END_LAT, END_LON = (
                ox.geocode(
                    end_search
                )
            )

        except Exception:

            return jsonify({

                "success":
                    False,

                "message":
                    f"도착지 '{end_query}'를 찾지 못했습니다."

            }), 400


        # ====================================================
        # 실제 보행 Node
        # ====================================================

        start_node = (
            ox.distance.nearest_nodes(
                G,
                X=START_LON,
                Y=START_LAT
            )
        )


        end_node = (
            ox.distance.nearest_nodes(
                G,
                X=END_LON,
                Y=END_LAT
            )
        )


        # ====================================================
        # 일반 최단 경로
        # ====================================================

        route_shortest = (
            nx.shortest_path(
                G,
                source=
                    start_node,
                target=
                    end_node,
                weight=
                    "length"
            )
        )


        # ====================================================
        # WalkAble 추천 경로
        # ====================================================

        route_accessible = (
            nx.shortest_path(
                G,
                source=
                    start_node,
                target=
                    end_node,
                weight=
                    "cost"
            )
        )


        shortest_data = (
            build_route_data(
                G,
                route_shortest,
                "length",
                "shortest"
            )
        )


        accessible_data = (
            build_route_data(
                G,
                route_accessible,
                "cost",
                "accessible"
            )
        )


        route_photos = (
            get_route_photos(
                accessible_data[
                    "edge_ids"
                ]
            )
        )


        # ====================================================
        # 두 경로 동일 여부
        # ====================================================

        same_route = (
            route_shortest
            ==
            route_accessible
        )


        accessible_data.pop(
            "edge_ids",
            None
        )

        shortest_data.pop(
            "edge_ids",
            None
        )


        # ====================================================
        # ★ 실제 경로 Node 위치를 반환
        #
        # 검색 좌표가 아니라
        # 경로가 실제 시작되는 Node 좌표
        # ====================================================

        START_NODE_LAT = float(
            G.nodes[
                start_node
            ][
                "y"
            ]
        )

        START_NODE_LON = float(
            G.nodes[
                start_node
            ][
                "x"
            ]
        )


        END_NODE_LAT = float(
            G.nodes[
                end_node
            ][
                "y"
            ]
        )

        END_NODE_LON = float(
            G.nodes[
                end_node
            ][
                "x"
            ]
        )


        print(
            "✅ 추천 경로 거리:",
            accessible_data[
                "distance"
            ]
        )

        print(
            "✅ 일반 경로 거리:",
            shortest_data[
                "distance"
            ]
        )

        print(
            "두 경로 동일:",
            same_route
        )

        print("=" * 70)


        return jsonify({

            "success":
                True,

            "same_route":
                same_route,

            "start": {

                "name":
                    start_query,

                "lat":
                    START_NODE_LAT,

                "lon":
                    START_NODE_LON

            },

            "end": {

                "name":
                    end_query,

                "lat":
                    END_NODE_LAT,

                "lon":
                    END_NODE_LON

            },

            "recommended":
                accessible_data,

            "shortest":
                shortest_data,

            "photos":
                route_photos

        })


    except nx.NetworkXNoPath:

        return jsonify({

            "success":
                False,

            "message":
                "출발지와 도착지 사이의 보행 경로를 찾지 못했습니다."

        }), 400


    except Exception as e:

        print(
            "❌ 경로 계산 오류:",
            repr(e)
        )


        return jsonify({

            "success":
                False,

            "message":
                "경로를 계산하는 중 오류가 발생했습니다.",

            "detail":
                str(e)

        }), 500


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )