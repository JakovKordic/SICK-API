from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)


def get_vehicle_history(obj_id: str, current_time: str):
    query = text("""
        WITH params AS (
            SELECT
                CAST(:current_time AS TIMESTAMP) AS current_time,
                CAST(:current_time AS TIMESTAMP) - INTERVAL '5 minutes' AS window_start
        ),
        context_row AS (
            SELECT
                obj_id,
                time,
                fuel_lvl,
                speed,
                contact_value,
                gps_latitude,
                gps_longitude
            FROM fuel_theft.vehicle_measurements, params
            WHERE obj_id = :obj_id
              AND time < params.window_start
            ORDER BY time DESC
            LIMIT 1
        ),
        window_rows AS (
            SELECT
                obj_id,
                time,
                fuel_lvl,
                speed,
                contact_value,
                gps_latitude,
                gps_longitude
            FROM fuel_theft.vehicle_measurements, params
            WHERE obj_id = :obj_id
              AND time < params.current_time
              AND time >= params.window_start
        )
        SELECT *
        FROM (
            SELECT * FROM context_row
            UNION ALL
            SELECT * FROM window_rows
        ) rows
        ORDER BY time ASC
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {
                "obj_id": str(obj_id),
                "current_time": current_time
            }
        ).mappings().all()

    return [dict(row) for row in rows]

def insert_measurement(measurement: dict):
    query = text("""
        INSERT INTO fuel_theft.vehicle_measurements (
            obj_id,
            time,
            fuel_lvl,
            speed,
            contact_value,
            gps_latitude,
            gps_longitude
        )
        VALUES (
            :obj_id,
            CAST(:time AS TIMESTAMP),
            :fuel_lvl,
            :speed,
            :contact_value,
            :gps_latitude,
            :gps_longitude
        )
    """)

    with engine.begin() as conn:
        conn.execute(query, measurement)