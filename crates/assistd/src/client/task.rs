//! `task` subcommands: list saved tasks, queue a run of one by name, or
//! cancel a run, all through the daemon.

use std::io::{self, Write};

use anyhow::{Result, bail};
use assistd_ipc::{Event, Request, TaskId, TaskRunId, TaskSpec};
use chrono::{DateTime, Local, Utc};
use clap::{Args, Subcommand};
use uuid::Uuid;

use super::run_one_shot;
use super::terminal_text::escape_controls_single_line;

#[derive(Args)]
pub(crate) struct TaskArgs {
    #[command(subcommand)]
    pub action: TaskAction,
}

#[derive(Subcommand)]
pub(crate) enum TaskAction {
    /// One tab-separated row per task: id, name, enabled or disabled, and when its
    /// triggers next fire in local time (`-` when nothing is due).
    List,
    /// Queue a run of the task named `name` now, whatever its triggers; for systemd
    /// timers, udev rules and key bindings. Prints the queued run's id.
    Run { name: String },
    /// Drop queued run `run`, or stop it if it is running.
    Cancel { run: i64 },
}

/// A task as `TaskList` reports it.
struct ListedTask {
    id: TaskId,
    spec: TaskSpec,
    next_fire: Option<DateTime<Utc>>,
}

pub(crate) async fn run(args: TaskArgs) -> Result<()> {
    match args.action {
        TaskAction::List => print_tasks().await,
        TaskAction::Run { name } => run_named_task(&name).await,
        TaskAction::Cancel { run } => cancel_run(TaskRunId(run)).await,
    }
}

async fn print_tasks() -> Result<()> {
    let mut out = io::stdout();
    for task in list_tasks().await? {
        let state = if task.spec.enabled {
            "enabled"
        } else {
            "disabled"
        };
        let next_fire = task.next_fire.map_or_else(
            || "-".to_string(),
            |at| {
                at.with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M %Z")
                    .to_string()
            },
        );
        let name = escape_controls_single_line(task.spec.name.as_str());
        writeln!(out, "{}\t{name}\t{state}\t{next_fire}", task.id)?;
    }
    Ok(())
}

async fn run_named_task(name: &str) -> Result<()> {
    let Some(task) = list_tasks()
        .await?
        .into_iter()
        .find(|task| task.spec.name.as_str() == name.trim())
    else {
        bail!("no task is named {:?}", escape_controls_single_line(name));
    };
    let request = Request::TaskRun {
        id: Uuid::new_v4().to_string(),
        task: task.id,
    };
    run_one_shot(request, |event| {
        if let Event::TaskRunQueued { run, .. } = event {
            writeln!(io::stdout(), "{run}")?;
        }
        Ok(())
    })
    .await
}

async fn cancel_run(run: TaskRunId) -> Result<()> {
    let request = Request::TaskCancel {
        id: Uuid::new_v4().to_string(),
        run,
    };
    run_one_shot(request, |_| Ok(())).await
}

async fn list_tasks() -> Result<Vec<ListedTask>> {
    let mut tasks = Vec::new();
    let request = Request::TaskList {
        id: Uuid::new_v4().to_string(),
    };
    run_one_shot(request, |event| {
        if let Event::Task {
            task_id,
            spec,
            next_fire,
            ..
        } = event
        {
            tasks.push(ListedTask {
                id: *task_id,
                spec: spec.clone(),
                next_fire: *next_fire,
            });
        }
        Ok(())
    })
    .await?;
    Ok(tasks)
}
