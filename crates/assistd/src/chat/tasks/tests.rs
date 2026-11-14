use super::*;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn deletes(panel: &mut TasksPanel) -> Vec<TaskId> {
    panel
        .take_requests()
        .into_iter()
        .filter_map(|request| match request.body {
            RequestBody::Task(TaskRequest::Delete { task }) => Some(task),
            _ => None,
        })
        .collect()
}

#[test]
fn an_undecodable_task_is_deleted_only_once_the_question_is_answered_with_enter() {
    let mut panel = TasksPanel::new(false, None);
    let list = panel.take_requests().remove(0).id;
    panel.on_event(Event::new(
        list.clone(),
        EventBody::UndecodableTask {
            task_id: TaskId(7),
            name: "digest".into(),
            reason: "unknown variant `hourly`".into(),
        },
    ));
    panel.on_event(Event::new(list, EventBody::Done));
    assert_eq!(panel.selected_row(), ListRow::Undecodable(0));

    panel.on_key(key(KeyCode::Enter));
    assert!(panel.editor().is_none());
    panel.on_key(key(KeyCode::Delete));
    panel.on_key(key(KeyCode::Esc));
    assert!(deletes(&mut panel).is_empty());
    assert!(panel.deleting().is_none());

    panel.on_key(key(KeyCode::Delete));
    panel.on_key(key(KeyCode::Enter));
    assert_eq!(deletes(&mut panel), [TaskId(7)]);
}
