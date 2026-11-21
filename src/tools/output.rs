//! What a tool produced.
//!
//! One type, because there is one answer to "what came back": the text the
//! model reads, plus any images that travel beside it.


/// What a tool produced.
///
/// Text is what the model reads. Images cannot travel in a tool message — the
/// OpenAI format restricts that content to text — so the loop collects them and
/// attaches them to a user message placed after the tool results.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// Text the model reads.
    pub text: String,
    /// Images the model should see.
    pub images: Vec<crate::message::ImageAttachment>,
}

impl ToolOutput {
    /// Text only.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Attach an image.
    pub fn with_image(mut self, image: crate::message::ImageAttachment) -> Self {
        self.images.push(image);
        self
    }

    /// Whether this output carries anything for the model to look at.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}
