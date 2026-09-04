"""
FastAPI backend entrypoint for Dam Break Inundation Simulation (SIH 26161).
Exposes /api/simulate to trigger a background simulation task.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field, field_validator
import uuid
import os
import json

app = FastAPI(
    title="Dam Break Inundation Simulation API",
    description="SIH 26161 - Backend for hydrodynamic dam-break simulation & hazard mapping",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store for prototype (replace with SQLite/Redis later if needed)
TASK_STORE = {}


class DamCoordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class SimulationRequest(BaseModel):
    dam_name: str = Field(..., min_length=1, max_length=200)
    coordinates: DamCoordinates
    crest_elevation: float = Field(..., gt=0, description="Crest elevation in meters (MSL)")
    breach_width: float = Field(..., gt=0, description="Breach width in meters")
    breach_formation_time: float = Field(default=1.0, gt=0, description="Breach formation time in hours")
    reservoir_volume: float | None = Field(default=None, gt=0, description="Reservoir volume in cubic meters")
    engine: str = Field(default="swe2d", description="Simulation engine: 'sph' or 'swe2d'")

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v):
        allowed = {"sph", "swe2d"}
        if v not in allowed:
            raise ValueError(f"engine must be one of {allowed}")
        return v


class SimulationResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None


def run_simulation_task(task_id: str, params: dict):
    """
    Background task orchestrating the hydro engine run and post-processing.
    Runs synchronously in a background thread (fine for prototype scale).
    """
    TASK_STORE[task_id] = {"status": "in_progress", "stage": "initializing", "progress": 5, "result": None}

    try:
        from services.breach_model import generate_breach_hydrograph
        from services.dem_processor import prepare_dem
        from services.converter import netcdf_to_hazard_vector

        TASK_STORE[task_id]["stage"] = "generating_breach_hydrograph"
        TASK_STORE[task_id]["progress"] = 20
        hydrograph = generate_breach_hydrograph(
            crest_elevation=params["crest_elevation"],
            breach_width=params["breach_width"],
            breach_formation_time=params["breach_formation_time"],
            reservoir_volume=params.get("reservoir_volume"),
        )

        TASK_STORE[task_id]["stage"] = "running_hydro_engine"
        TASK_STORE[task_id]["progress"] = 50

        if params["engine"] == "sph":
            from services.sph_solver import run_sph_engine
            nc_output_path = run_sph_engine(params, hydrograph)
        else:
            from services.swe_solver import run_swe2d_engine
            nc_output_path = run_swe2d_engine(params, hydrograph)

        TASK_STORE[task_id]["stage"] = "generating_hazard_vectors"
        TASK_STORE[task_id]["progress"] = 85
        hazard_outputs = netcdf_to_hazard_vector(nc_output_path, out_format="both")

        TASK_STORE[task_id]["status"] = "completed"
        TASK_STORE[task_id]["stage"] = "complete"
        TASK_STORE[task_id]["progress"] = 100
        TASK_STORE[task_id]["result"] = {
            "netcdf_path": nc_output_path,
            "hazard_geojson": hazard_outputs.get("geojson"),
            "hazard_kml": hazard_outputs.get("kml"),
        }

    except Exception as exc:
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["result"] = {"error": str(exc)}


@app.post("/api/simulate", response_model=SimulationResponse, status_code=status.HTTP_202_ACCEPTED)
async def simulate_dam_break(request: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Accepts dam parameters and queues a background simulation task.
    Returns a task_id for polling status via /api/simulate/status/{task_id}.
    """
    task_id = str(uuid.uuid4())

    params = {
        "dam_name": request.dam_name,
        "latitude": request.coordinates.latitude,
        "longitude": request.coordinates.longitude,
        "crest_elevation": request.crest_elevation,
        "breach_width": request.breach_width,
        "breach_formation_time": request.breach_formation_time,
        "reservoir_volume": request.reservoir_volume,
        "engine": request.engine,
    }

    background_tasks.add_task(run_simulation_task, task_id, params)

    return SimulationResponse(
        task_id=task_id,
        status="queued",
        message=f"Simulation for '{request.dam_name}' queued successfully using {request.engine} engine.",
    )


@app.get("/api/simulate/status/{task_id}", response_model=TaskStatusResponse)
async def get_simulation_status(task_id: str):
    """Polls in-memory task store for progress/result."""
    task = TASK_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
    )


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "dam-break-simulation-api"}
