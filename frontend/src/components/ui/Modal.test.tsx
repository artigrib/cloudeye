import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Modal from './Modal'

describe('Modal', () => {
  it('closes on scrim click but not on content click', () => {
    const onClose = vi.fn()
    render(
      <Modal onClose={onClose}>
        <p>dialog content</p>
      </Modal>,
    )
    fireEvent.click(screen.getByText('dialog content'))
    expect(onClose).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText('dialog content').closest('.fixed')!)
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('closes on Escape', () => {
    const onClose = vi.fn()
    render(
      <Modal onClose={onClose}>
        <p>dialog content</p>
      </Modal>,
    )
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()
  })
})
