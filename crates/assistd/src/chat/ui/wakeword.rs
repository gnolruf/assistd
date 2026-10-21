//! The `/wakeword` panel: the menu, the new-phrase input, and the listening
//! computer that enrollment and testing share.

use std::borrow::Cow;
use std::time::Instant;

use assistd_ipc::wakeword::MAX_WAKE_PHRASE_WORDS;
use assistd_ipc::{SampleRejection, WakePhraseText};
use assistd_voice::listen::wake_phrase::{DEFAULT_WAKE_PHRASE, PhraseStrength};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};
use textwrap::core::display_width;

use super::super::app::{
    ComputerScene, EnrollProgress, LANE_COLUMNS, MenuItem, Mood, RECENT_RESULTS, SampleResult,
    SessionPurpose, StatusState, TestProgress, WakeWordPanel, WakeWordStatus, WakeWordView,
};
use super::super::markdown::BLEY;
use super::{rule_style, saturating_u16, selected_style};
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
/// it holds still as they arrive: a prompt, a progress meter, a gap, then the results.
const CAPTION_ROWS: usize = 3 + RECENT_RESULTS;
const INPUT_MAX_COLS: u16 = 44;
/// Columns beside the input box kept for the strength badge.
const STRENGTH_COLS: u16 = 18;

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
        WakeWordView::Menu(selected) => {
            let body = menu_body(&panel.status, *selected);
            frame.render_widget(wrapped(body), inner);
        }
        WakeWordView::Update(phrase) => render_update(frame, inner, phrase),
        WakeWordView::Session(session) => {
            let captions = match &session.purpose {
                SessionPurpose::Enroll(progress) => enroll_captions(progress),
                SessionPurpose::Test(progress) => test_captions(progress),
            };
            render_scene(frame, inner, &session.scene, captions);
        }
        WakeWordView::Saved { phrase, variants } => {
            frame.render_widget(wrapped(saved_body(phrase, variants)), inner);
        }
    }
}

/// What the status line suggests while the panel is open.
pub(super) fn wake_word_hint(view: &WakeWordView) -> &'static str {
    match view {
        WakeWordView::Menu(_) => "↑/↓ select · Enter choose · Esc close",
        WakeWordView::Update(_) => "Enter record · Ctrl+U clear · Esc back",
        WakeWordView::Session(session) => match session.purpose {
            SessionPurpose::Enroll(_) => "Esc cancel",
            SessionPurpose::Test(_) => "Esc stop",
        },
        WakeWordView::Saved { .. } => "t test · Enter back",
    }
}

fn panel_title(view: &WakeWordView) -> &'static str {
    match view {
        WakeWordView::Menu(_) => " Wake word ",
        WakeWordView::Update(_) => " New wake word ",
        WakeWordView::Session(session) => match session.purpose {
            SessionPurpose::Enroll(_) => " Learning your wake word ",
            SessionPurpose::Test(_) => " Testing your wake word ",
        },
        WakeWordView::Saved { .. } => " Wake word updated ",
    }
}

fn menu_body(status: &StatusState, selected: MenuItem) -> Vec<Line<'static>> {
    let mut body = status_rows(status);
    body.push(Line::default());
    body.extend(MenuItem::ALL.into_iter().map(|item| {
        let (name, about) = menu_text(item);
        let (marker, style) = if item == selected {
            ("› ", selected_style())
        } else {
            ("  ", Style::default())
        };
        Line::from(vec![
            Span::styled(format!("{marker}{name:<8}"), style),
            Span::raw("  "),
            Span::styled(about, dim()),
        ])
    }));
    body
}

fn menu_text(item: MenuItem) -> (&'static str, String) {
    match item {
        MenuItem::Update => ("Update", "Record a new wake word".to_string()),
        MenuItem::Test => ("Test", "Say it and see whether I'd wake".to_string()),
        MenuItem::Reset => ("Reset", format!("Go back to \"{DEFAULT_WAKE_PHRASE}\"")),
    }
}

fn status_rows(status: &StatusState) -> Vec<Line<'static>> {
    match status {
        StatusState::Loading => vec![Line::styled("Reading the saved wake word…", dim())],
        StatusState::Failed(message) => vec![Line::styled(
            format!("Couldn't read the wake word: {}", escaped(message)),
            bad(),
        )],
        StatusState::Loaded(status) => loaded_status_rows(status),
    }
}

fn loaded_status_rows(status: &WakeWordStatus) -> Vec<Line<'static>> {
    let source = match status.samples {
        0 => "built in".to_string(),
        samples => format!("learned from {samples} samples"),
    };
    let variants = if status.variants.is_empty() {
        "nothing else yet".to_string()
    } else {
        escaped(&status.variants.join(", ")).into_owned()
    };
    let listening = if status.listening { "on" } else { "off" };
    vec![
        labelled(
            "Wake word",
            vec![
                Span::styled(format!("\"{}\"", escaped(&status.phrase)), bold()),
                Span::styled(format!("  ({source})"), dim()),
            ],
        ),
        labelled("Also hears", vec![Span::raw(variants)]),
        labelled("Listening", vec![Span::raw(listening)]),
    ]
}

fn labelled(label: &str, mut value: Vec<Span<'static>>) -> Line<'static> {
    value.insert(0, Span::styled(format!("{label:<12}"), dim()));
    Line::from(value)
}

/// The prompt, the input box with the strength badge beside it, then a tip
/// or the reason the phrase is refused.
fn render_update(frame: &mut Frame<'_>, area: Rect, phrase: &str) {
    let [prompt_area, _, input_row, _, tip_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new("Type a new wake word, then press Enter to record it."),
        prompt_area,
    );
    let box_width = area
        .width
        .saturating_sub(STRENGTH_COLS)
        .clamp(1, INPUT_MAX_COLS);
    let [input_area, _, strength_area] = Layout::horizontal([
        Constraint::Length(box_width),
        Constraint::Length(2),
        Constraint::Min(0),
    ])
    .areas(input_row);
    render_phrase_input(frame, input_area, phrase);
    let checked = (!phrase.trim().is_empty()).then(|| WakePhraseText::try_from(phrase.to_string()));
    if let Some(Ok(valid)) = &checked {
        let badge_area = Rect {
            y: strength_area.y.saturating_add(1),
            height: 1,
            ..strength_area
        };
        frame.render_widget(
            Paragraph::new(strength_badge(PhraseStrength::of(valid.as_str()))),
            badge_area,
        );
    }
    let tip = match checked {
        Some(Err(refused)) => Line::styled(format!("Can't use that: {refused}."), bad()),
        Some(Ok(_)) | None => Line::styled(
            format!(
                "Up to {MAX_WAKE_PHRASE_WORDS} words. Longer phrases wake by accident less \
                 often: aim for 4 or more syllables, like \"{DEFAULT_WAKE_PHRASE}\"."
            ),
            dim(),
        ),
    };
    frame.render_widget(wrapped(vec![tip]), tip_area);
}

/// The phrase in a rounded box, scrolled to keep its end and the cursor in view.
fn render_phrase_input(frame: &mut Frame<'_>, area: Rect, phrase: &str) {
    let phrase = escape_controls_single_line(phrase);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(rule_style())
        .padding(Padding::horizontal(1));
    let inner = block.inner(area);
    let typed = saturating_u16(display_width(&phrase));
    let scroll = typed.saturating_sub(inner.width.saturating_sub(1));
    let text = if phrase.is_empty() {
        Span::styled("hey jarvis", Style::default().fg(Color::DarkGray))
    } else {
        Span::raw(phrase.into_owned())
    };
    frame.render_widget(Paragraph::new(text).scroll((0, scroll)).block(block), area);
    if inner.width > 0 && inner.height > 0 {
        let cursor_x = inner.x.saturating_add(typed - scroll);
        frame.set_cursor_position(Position::new(cursor_x, inner.y));
    }
}

fn strength_badge(strength: PhraseStrength) -> Line<'static> {
    let (label, color) = match strength {
        PhraseStrength::Low => ("low", Color::Red),
        PhraseStrength::Medium => ("medium", Color::Yellow),
        PhraseStrength::High => ("high", Color::Green),
    };
    Line::from(vec![
        Span::styled("strength ", dim()),
        Span::styled(
            format!("● {label}"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ])
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

fn enroll_captions(progress: &EnrollProgress) -> Vec<Line<'static>> {
    let phrase = escaped(&progress.phrase);
    let mut captions = vec![
        Line::styled(
            format!(
                "Say \"{phrase}\" until I've heard it clearly {} times.",
                progress.required
            ),
            bold(),
        ),
        progress_meter(progress.confident, progress.required),
        Line::default(),
    ];
    captions.extend(progress.results.iter().map(sample_line));
    captions
}

fn progress_meter(confident: u8, required: u8) -> Line<'static> {
    let filled = "●".repeat(usize::from(confident.min(required)));
    let empty = "○".repeat(usize::from(required.saturating_sub(confident)));
    Line::from(vec![
        Span::styled(filled, good()),
        Span::styled(empty, dim()),
        Span::styled(format!("  {confident} of {required} clear"), dim()),
    ])
}

fn sample_line(result: &SampleResult) -> Line<'static> {
    match result {
        SampleResult::Confident { heard } => {
            Line::styled(format!("✓ heard \"{}\"", escaped(heard)), good())
        }
        SampleResult::Variant { heard } => Line::styled(
            format!("~ heard \"{}\", kept as a variant", escaped(heard)),
            Style::default().fg(Color::Yellow),
        ),
        SampleResult::Rejected { reason, heard } => {
            let heard = heard.as_deref().map_or_else(String::new, |heard| {
                format!(": heard \"{}\"", escaped(heard))
            });
            Line::styled(format!("✗ {}{heard}", rejection_text(*reason)), bad())
        }
    }
}

fn rejection_text(reason: SampleRejection) -> &'static str {
    match reason {
        SampleRejection::TooShort => "too short; say the whole phrase",
        SampleRejection::TooLong => "too long; say just the phrase",
        SampleRejection::Clipped => "clipped; turn the mic gain down",
        SampleRejection::DidNotMatch => "didn't sound like the phrase",
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
    captions.extend(progress.results.iter().map(|result| {
        let heard = escaped(&result.heard);
        if result.matched {
            Line::styled(format!("✓ would wake: \"{heard}\""), good())
        } else {
            Line::styled(format!("✗ would not wake: \"{heard}\""), bad())
        }
    }));
    captions
}

fn saved_body(phrase: &str, variants: &[String]) -> Vec<Line<'static>> {
    let mut body = vec![Line::styled(
        format!("✓ Wake word updated to \"{}\"", escaped(phrase)),
        good().add_modifier(Modifier::BOLD),
    )];
    if !variants.is_empty() {
        body.push(Line::raw(format!(
            "  Also hears: {}",
            escaped(&variants.join(", "))
        )));
    }
    body.push(Line::default());
    body.push(Line::styled(
        "Press t to test it, or Enter to go back.",
        dim(),
    ));
    body
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
