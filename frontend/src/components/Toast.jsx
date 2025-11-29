export default function Toast({ message }) {
  if (!message) return null
  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-emerald-600 text-white px-4 py-2 rounded-lg shadow-lg">
        {message}
      </div>
    </div>
  )
}
