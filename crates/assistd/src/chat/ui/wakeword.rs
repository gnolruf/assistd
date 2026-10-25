//! The `/wakeword` panel: the configured phrase, and the listening computer
//! a test session draws.

use std::borrow::Cow;
use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};

use super::super::app::{
    ComputerScene, LANE_COLUMNS, Mood, RECENT_RESULTS, StatusState, TestProgress, WakeWordPanel,
    WakeWordStatus, WakeWordView,
};
use super::super::markdown::BLEY;
use super::saturating_u16;
use crate::client::terminal_text::escape_controls_single_line;

const COMPUTER: [&str; 23] = [
    r" ______________________________",
    r"|\                             \",
    r"| \                             \",
    r"|  \_____________________________\",
    r"|  |  _________________________  |",
    r"|  | |                         | |",
    r"|  | |                         | |",
    r"|  | |       █         █       | |",
    r"|  | |                         | |",
    r"|  | |      \           /      | |",
    r"|  | |       '-._____.-'       | |",
    r"|  | |                         | |",
    r"|  | |_________________________| |",
    r"|  |                             |",
    r"|  |                ________     |",
    r"\  |  ::            --------     |",
    r" \ |_____________________________|",
    r"  \|_____________________________|",
    r"    _______________________________",
    r"   \ [][][][][][][][][][][][][][]  \",
    r"    \ [][][][][][][][][][][][][][]  \",
    r"     \ [][] [________________] [][]  \",
    r"      \_______________________________\",
];

/// First row of [`COMPUTER`]'s face, from the brows down to the mouth; the
/// content face is the one drawn there.
const FACE_TOP_ROW: usize = 5;
const FACE_ROWS: usize = 6;

const WONDERING_FACE: [&str; FACE_ROWS] = [
    r"|  | |                ,-.      | |",
    r"|  | |      ---                | |",
    r"|  | |       █         █       | |",
    r"|  | |                         | |",
    r"|  | |                         | |",
    r"|  | |        ─────────        | |",
];

const WORRIED_FACE: [&str; FACE_ROWS] = [
    r"|  | |                         | |",
    r"|  | |      _/        \_       | |",
    r"|  | |       █         █       | |",
    r"|  | |                         | |",
    r"|  | |                         | |",
    r"|  | |        /\/\/\/\/        | |",
];

/// Row of [`COMPUTER`] the lanes run along: the middle of its screen.
const LANE_ROW: usize = 8;
/// Rows kept between the computer and the captions under it.
const SCENE_GAP_ROWS: usize = 1;
/// Caption rows kept under the computer however many results there are, so
/// it holds still as they arrive: a prompt, a gap, then the results.
const CAPTION_ROWS: usize = 2 + RECENT_RESULTS;

/// Draw the panel over `area`, in place of the transcript and input line.
pub(super) fn render_wake_word_panel(frame: &mut Frame<'_>, area: Rect, panel: &WakeWordPanel) {
    let accent = Style::default().fg(BLEY);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .padding(Padding::horizontal(1))
        .title(Span::styled(
            panel_title(&panel.view),
            accent.add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    match &panel.view {
        WakeWordView::Status => frame.render_widget(wrapped(status_body(&panel.status)), inner),
        WakeWordView::Session(session) => {
            render_scene(
                frame,
                inner,
                &session.scene,
                test_captions(&session.progress),
            );
        }
    }
}

/// What the status line suggests while the panel is open.
pub(super) fn wake_word_hint(view: &WakeWordView) -> &'static str {
    match view {
        WakeWordView::Status => "Enter test · Esc close",
        WakeWordView::Session(_) => "Esc stop",
    }
}

fn panel_title(view: &WakeWordView) -> &'static str {
    match view {
        WakeWordView::Status => " Wake word ",
        WakeWordView::Session(_) => " Testing your wake word ",
    }
}

/// The phrase and whether listening is on, then how to change or test it.
fn status_body(status: &StatusState) -> Vec<Line<'static>> {
    let mut body = status_rows(status);
    body.extend([
        Line::default(),
        Line::styled(
            "Change it as voice.wakeword.phrase in /settings, then restart the daemon.",
            dim(),
        ),
        Line::styled("Press Enter to say it and see whether I'd wake.", dim()),
    ]);
    body
}

fn status_rows(status: &StatusState) -> Vec<Line<'static>> {
    match status {
        StatusState::Loading => vec![Line::styled("Reading the wake word…", dim())],
        StatusState::Failed(message) => vec![Line::styled(
            format!("Couldn't read the wake word: {}", escaped(message)),
            bad(),
        )],
        StatusState::Loaded(status) => loaded_status_rows(status),
    }
}

fn loaded_status_rows(status: &WakeWordStatus) -> Vec<Line<'static>> {
    let listening = if status.listening { "on" } else { "off" };
    vec![
        labelled(
            "Wake word",
            vec![Span::styled(
                format!("\"{}\"", escaped(&status.phrase)),
                bold(),
            )],
        ),
        labelled("Listening", vec![Span::raw(listening)]),
    ]
}

fn labelled(label: &str, mut value: Vec<Span<'static>>) -> Line<'static> {
    value.insert(0, Span::styled(format!("{label:<12}"), dim()));
    Line::from(value)
}

/// The computer above `captions`, the two centred together in `area`; the
/// computer is left out when the panel is too short for both.
fn render_scene(
    frame: &mut Frame<'_>,
    area: Rect,
    scene: &ComputerScene,
    captions: Vec<Line<'static>>,
) {
    let scene_rows = COMPUTER.len() + SCENE_GAP_ROWS;
    let height = usize::from(area.height);
    if height < scene_rows + captions.len() {
        frame.render_widget(wrapped(captions).alignment(Alignment::Center), area);
        return;
    }
    let top = height.saturating_sub(scene_rows + CAPTION_ROWS.max(captions.len())) / 2;
    let [_, scene_area, caption_area] = Layout::vertical([
        Constraint::Length(saturating_u16(top)),
        Constraint::Length(saturating_u16(scene_rows)),
        Constraint::Min(0),
    ])
    .areas(area);
    render_computer(frame, scene_area, scene);
    frame.render_widget(wrapped(captions).alignment(Alignment::Center), caption_area);
}

/// The computer centred across `area`, with as much of the inbound lane as
/// fits to its left and the outbound lane clipped at the right edge.
fn render_computer(frame: &mut Frame<'_>, area: Rect, scene: &ComputerScene) {
    let width = saturating_u16(computer_width());
    let art_x = area.x + area.width.saturating_sub(width) / 2;
    let lane_cols = usize::from(art_x - area.x).min(LANE_COLUMNS);
    let x = art_x - saturating_u16(lane_cols);
    let lanes_area = Rect {
        x,
        width: area.right() - x,
        ..area
    };
    let lines = computer_lines(scene, scene.mood(Instant::now()), lane_cols);
    frame.render_widget(Paragraph::new(lines), lanes_area);
}

fn computer_width() -> usize {
    COMPUTER
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0)
}

/// Each row of the computer wearing `mood`'s face, behind the last
/// `lane_cols` of the inbound lane, the lane row followed by the outbound one.
fn computer_lines(scene: &ComputerScene, mood: Mood, lane_cols: usize) -> Text<'static> {
    let face = face_rows(mood);
    let quiet_lane = " ".repeat(lane_cols);
    let inbound: String = scene
        .inbound()
        .chars()
        .skip(LANE_COLUMNS - lane_cols)
        .collect();
    let lines = COMPUTER.iter().enumerate().map(|(row, art)| {
        let art = row
            .checked_sub(FACE_TOP_ROW)
            .and_then(|face_row| face.get(face_row))
            .unwrap_or(art);
        if row == LANE_ROW {
            Line::from(vec![
                Span::styled(inbound.clone(), Style::default().fg(BLEY)),
                Span::raw(*art),
                Span::styled(
                    scene.outbound(),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
            ])
        } else {
            Line::from(vec![Span::raw(quiet_lane.clone()), Span::raw(*art)])
        }
    });
    Text::from(lines.collect::<Vec<_>>())
}

fn face_rows(mood: Mood) -> &'static [&'static str] {
    match mood {
        Mood::Content => &COMPUTER[FACE_TOP_ROW..FACE_TOP_ROW + FACE_ROWS],
        Mood::Wondering => &WONDERING_FACE,
        Mood::Worried => &WORRIED_FACE,
    }
}

fn test_captions(progress: &TestProgress) -> Vec<Line<'static>> {
    let ask = progress.phrase.as_deref().map_or_else(
        || "Say the wake word and I'll tell you whether I'd wake.".to_string(),
        |phrase| {
            format!(
                "Say \"{}\" and I'll tell you whether I'd wake.",
                escaped(phrase)
            )
        },
    );
    let mut captions = vec![Line::styled(ask, bold()), Line::default()];
    if progress.results.is_empty() {
        captions.push(Line::styled("Listening…", dim()));
    }
    captions.extend(progress.results.iter().map(|&matched| {
        if matched {
            Line::styled("✓ heard the wake word: I'd wake", good())
        } else {
            Line::styled("✗ heard speech without the wake word", bad())
        }
    }));
    captions
}

fn wrapped(lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false })
}

fn escaped(text: &str) -> Cow<'_, str> {
    escape_controls_single_line(text)
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn dim() -> Style {
    Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)
}

fn good() -> Style {
    Style::default().fg(Color::Green)
}

fn bad() -> Style {
    Style::default().fg(Color::Red)
}
