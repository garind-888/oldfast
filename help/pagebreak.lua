-- Pagebreak filter for Pandoc → DOCX (and other formats)
-- Converts
--   * raw  \newpage  or  \pagebreak
--   * a horizontal rule or empty div with class "pagebreak"
-- into a hard page‑break in the target format.

local pagebreak_openxml = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

-- raw TeX blocks (\newpage or \pagebreak)
function RawBlock (el)
  if el.text == '\\newpage' or el.text == '\\pagebreak' then
    return pandoc.RawBlock('openxml', pagebreak_openxml)
  end
end

-- --- or *** rule with class .pagebreak
function HorizontalRule ()
  return pandoc.RawBlock('openxml', pagebreak_openxml)
end

-- empty div of class .pagebreak
function Div (el)
  if el.classes:includes('pagebreak') then
    return pandoc.RawBlock('openxml', pagebreak_openxml)
  end
end