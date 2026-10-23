//! `#[derive(DescribeSettings)]`, which reads each config field's
//! `#[setting(...)]` into the metadata the settings panel shows.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::meta::ParseNestedMeta;
use syn::{Data, DeriveInput, Error, Field, Fields, LitStr, Token, parse_macro_input};

/// What a field's `#[setting(...)]` declares.
enum FieldSetting {
    Key {
        label: LitStr,
        description: LitStr,
        advanced: bool,
    },
    Nested,
}

/// The parts of a field's `#[serde(...)]` that decide its key.
#[derive(Default)]
struct SerdeField {
    skip: bool,
    rename: Option<LitStr>,
}

/// Implements `DescribeSettings` from each named field's `#[setting(label =
/// "..", description = "..", advanced)]`, or `#[setting(nested)]` for a table.
/// Fields serde skips are left out; any other field without one fails to compile.
#[proc_macro_derive(DescribeSettings, attributes(setting))]
pub fn derive_describe_settings(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(input, "DescribeSettings needs a struct"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new_spanned(
            input,
            "DescribeSettings needs named fields",
        ));
    };
    let steps = fields
        .named
        .iter()
        .filter_map(|field| field_step(field).transpose())
        .collect::<syn::Result<Vec<_>>>()?;
    let name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::assistd_config::settings::DescribeSettings
            for #name #type_generics #where_clause
        {
            fn describe_settings(
                prefix: &str,
                out: &mut ::std::vec::Vec<::assistd_config::settings::SettingInfo>,
            ) {
                #(#steps)*
            }
        }
    })
}

/// The statement that describes `field`, or `None` when serde skips it.
fn field_step(field: &Field) -> syn::Result<Option<TokenStream2>> {
    let serde = serde_field(field)?;
    if serde.skip {
        return Ok(None);
    }
    let Some(ident) = &field.ident else {
        return Err(Error::new_spanned(
            field,
            "DescribeSettings needs named fields",
        ));
    };
    let name = serde
        .rename
        .map_or_else(|| ident.to_string(), |rename| rename.value());
    let key = quote! { ::assistd_config::settings::child_key(prefix, #name) };
    let step = match field_setting(field)? {
        FieldSetting::Nested => {
            let ty = &field.ty;
            quote! {
                <#ty as ::assistd_config::settings::DescribeSettings>::describe_settings(&#key, out);
            }
        }
        FieldSetting::Key {
            label,
            description,
            advanced,
        } => quote! {
            out.push(::assistd_config::settings::SettingInfo {
                key: #key,
                label: #label,
                description: #description,
                advanced: #advanced,
            });
        },
    };
    Ok(Some(step))
}

fn field_setting(field: &Field) -> syn::Result<FieldSetting> {
    let Some(attr) = field
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("setting"))
    else {
        return Err(Error::new_spanned(
            field,
            "config field needs #[setting(label = \"..\", description = \"..\")] \
             or #[setting(nested)]",
        ));
    };
    let mut label = None;
    let mut description = None;
    let mut advanced = false;
    let mut nested = false;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("label") {
            label = Some(meta.value()?.parse()?);
        } else if meta.path.is_ident("description") {
            description = Some(meta.value()?.parse()?);
        } else if meta.path.is_ident("advanced") {
            advanced = true;
        } else if meta.path.is_ident("nested") {
            nested = true;
        } else {
            return Err(meta.error("expected label, description, advanced or nested"));
        }
        Ok(())
    })?;
    match (nested, label, description) {
        (true, None, None) if !advanced => Ok(FieldSetting::Nested),
        (false, Some(label), Some(description)) => Ok(FieldSetting::Key {
            label,
            description,
            advanced,
        }),
        _ => Err(Error::new_spanned(
            attr,
            "expected label and description (optionally advanced), or nested alone",
        )),
    }
}

fn serde_field(field: &Field) -> syn::Result<SerdeField> {
    let mut serde = SerdeField::default();
    for attr in field
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("serde"))
    {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                serde.skip = true;
            } else if meta.path.is_ident("rename") {
                serde.rename = Some(meta.value()?.parse()?);
            } else {
                skip_meta_value(&meta)?;
            }
            Ok(())
        })?;
    }
    Ok(serde)
}

/// Consumes the `= value` or `(...)` after a serde option this macro ignores.
fn skip_meta_value(meta: &ParseNestedMeta<'_>) -> syn::Result<()> {
    if meta.input.peek(Token![=]) {
        meta.value()?.parse::<syn::Expr>()?;
    } else if meta.input.peek(syn::token::Paren) {
        meta.parse_nested_meta(|inner| skip_meta_value(&inner))?;
    }
    Ok(())
}
